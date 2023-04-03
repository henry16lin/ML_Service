import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score, recall_score, precision_score
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')


class nn_factory():
    
    def __init__(self, model, device, X_df, y_df, batch_size, model_save_path):
        
        self.model = model.to(device)
        self.device = device
        self.model_save_path = model_save_path
        self.threshold = 0.5
        
        # split validation set
        X_train, X_val, y_train, y_val = train_test_split(X_df, y_df, test_size=0.2, random_state=33)
        
        self.X_train_tensor = torch.from_numpy(np.array(X_train)).to(self.device)
        self.y_train_tensor = torch.from_numpy(np.array(y_train)).to(self.device)
        self.X_val_tensor = torch.from_numpy(np.array(X_val)).to(self.device)
        self.y_val_tensor = torch.from_numpy(np.array(y_val)).to(self.device)
        
        
        self.train_loader = DataLoader(TensorDataset(self.X_train_tensor, self.y_train_tensor), 
                                  batch_size = batch_size, shuffle = True)
        
        self.val_loader = DataLoader(TensorDataset(self.X_val_tensor, self.y_val_tensor), 
                                  batch_size = batch_size, shuffle = False)
    
    
        cat = list(set(y_train))
        nSamples = [sum(y_train==c) for c in cat]
        self.class_weights = [1 - (x / sum(nSamples)) for x in nSamples]
    

    def fit(self, epoch):
        # optimizer = optim.Adam(self.model.parameters())
        optimizer = optim.AdamW(self.model.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)

        best_val_loss = 10000000
        best_val_acc = 0
        train_loss_hist, train_acc_hist, train_recall_hist, train_precision_hist = [], [], [], []
        val_loss_hist, val_acc_hist, val_recall_hist, val_precision_hist = [], [], [], []
        
        for ep in range(1, epoch + 1):
            epoch_begin = time.time()
            train_loss, train_acc, train_recall, train_precision = self.train(optimizer, ep)
            val_loss, val_acc, val_recall, val_precision = self.val()
            
            scheduler.step()
            print('elapse: %.2fs \n' % (time.time() - epoch_begin))
    
            if val_loss <= best_val_loss:
                print('improve validataion loss, saving model...\n')
                torch.save(self.model.state_dict(),
                           os.path.join(self.model_save_path, 'epoch-%d-val_loss%.3f-val_acc%.3f.pt'
                           % (ep, val_loss, val_acc)))
                
                best_val_loss = val_loss
                best_val_acc = val_acc
    
            train_loss_hist.append(train_loss)
            train_acc_hist.append(train_acc)
            train_recall_hist.append(train_recall)
            train_precision_hist.append(train_precision)
            val_loss_hist.append(val_loss)
            val_acc_hist.append(val_acc)
            val_recall_hist.append(val_recall)
            val_precision_hist.append(val_precision)

        #save final model
        # fast but also need to save out dimension of each layer or Net class
        state = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': optimizer.state_dict()
                }
        torch.save(state, os.path.join(self.model_save_path, 'last_model.pt'))
        
        ### graph train hist ###
        self.graph_hist(loss={'train':train_loss_hist, 'val': val_loss_hist}, 
                        acc={'train': train_acc_hist, 'val': val_acc_hist},
                        recall={'train': train_recall_hist, 'val': val_recall_hist},
                        precision={'train': train_precision_hist, 'val': val_precision_hist}
                        )
    
    def predict_proba(self, df):
        
        #np_data = self.transformer.transform(df)
        tensor_data = torch.from_numpy(np.array(df)).to(self.device)

        with torch.no_grad():
            log_prob = F.log_softmax(self.model(tensor_data.float()))
            pred_prob = torch.exp(log_prob).data.cpu().numpy()
        
        return pred_prob
    
    
    def predict(self, df):
        pred_prob = self.predict_proba(df)
        pred_ind = np.argmax(pred_prob, axis=1)
        
        return pred_ind
    
    
    def train(self, optimizer, epoch):
        
        device = self.device
        train_loader = self.train_loader
        
        print('[epoch %d]train on %d data......' % (epoch,len(train_loader.dataset)))
        train_loss = 0
        train_pred, train_target = [], []
        self.model.train()
        for batch_ind, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = self.model(data.float())
            
            weights =  torch.tensor(self.class_weights).to(device)
            # criterion = FocalLoss(weight=weights)
            criterion = nn.CrossEntropyLoss(weight=weights)
            loss = criterion(output, target)
    
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            log_prob = F.log_softmax(output, dim=1)
            pred_prob = torch.exp(log_prob).data.cpu().numpy()[:, 1]
            train_pred.extend((pred_prob >= self.threshold).astype(int))
            train_target.extend(target.cpu().detach().numpy().tolist())
    
        train_loss /= len(train_loader.dataset)
        acc = accuracy_score(train_target, train_pred)
        recall = recall_score(train_target, train_pred)
        precision = precision_score(train_target, train_pred)
    
        print('training set: average loss:%.4f, acc:%.3f, recall: %.3f, precission: %.3f'\
              %(train_loss, 100 * acc, recall, precision))
    
        return train_loss, acc, recall, precision
    
    
    def val(self):
        model = self.model
        device = self.device
        val_loader = self.val_loader
        
        print('validation on %d data......' % len(val_loader.dataset))
        model.eval()
        val_loss = 0
        val_pred, val_target = [], []
        with torch.no_grad(): #temporarily set all the requires_grad flag to false
            for data,target in val_loader:
                data,target = data.to(device),target.to(device)
                output = model(data.float())
                
                weights =  torch.tensor(self.class_weights).to(device)
                # criterion = FocalLoss(weight=weights)
                criterion = nn.CrossEntropyLoss(weight=weights)
                val_loss += criterion(output, target).item() #sum up batch loss
    
                log_prob = F.log_softmax(output, dim=1)
                val_prob = torch.exp(log_prob).data.cpu().numpy()[:, 1]
                val_pred.extend((val_prob > self.threshold).astype(int))
                val_target.extend(target.cpu().detach().numpy().tolist())
            
            val_loss /= len(val_loader.dataset) #avg of sum of batch loss
            acc = accuracy_score(val_target, val_pred)
            recall = recall_score(val_target, val_pred)
            precision = precision_score(val_target, val_pred)
    
        print('Val set: average loss:%.4f, acc:%.3f, recall: %.3f, precission: %.3f'\
              %(val_loss, 100 * acc, recall, precision))
    
        return val_loss, acc, recall, precision
    
    def graph_hist(self, **kwargs):
        i = 1
        plt.figure(figsize=(24, 6))
        for k1, v1 in kwargs.items():
            plt.subplot(1, len(kwargs), i)
            for k2, v2 in v1.items():
                plt.plot(v2)
            plt.legend(['train %s' % k1, 'val %s' % k1], loc='best')
            i += 1
        plt.savefig(os.path.join(self.model_save_path, 'train_hist.jpg'))
        plt.close()
        

# since i don't want boosting user need to install pytorch so i didn't put this loss fn in loss.py
class FocalLoss(nn.Module):
    def __init__(self, weight, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss
    
