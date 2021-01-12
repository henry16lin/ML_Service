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
import numpy as np



cwd = os.getcwd()
model_save_path = os.path.join(cwd,'checkpoint')

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)


class nn_factory():
    
    def __init__(self, model, device, X_df, y_df, batch_size):
        
        self.model = model.to(device)
        self.device = device
        
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
        
        optimizer = optim.Adam(self.model.parameters())
        val_loss = 10000000
        val_acc = 0
        train_loss_hist,train_acc_hist = [],[]
        val_loss_hist,val_acc_hist = [],[]
        
        for ep in range(1, epoch + 1):
            epoch_begin = time.time()
            cur_train_loss,cur_train_acc = self.train(optimizer,ep)
            cur_val_loss,cur_val_acc = self.val()
            
            print('elapse:%.2fs \n'%(time.time()-epoch_begin))
    
            if cur_val_loss<=val_loss:
                print('improve validataion loss, saving model...\n')
                torch.save(self.model.state_dict(), os.path.join(model_save_path,'epoch-%d-val_loss%.3f-val_acc%.3f.pt' %(ep,cur_val_loss,cur_val_acc) ))
                
                val_loss = cur_val_loss
                val_acc = cur_val_acc
    
            train_loss_hist.append(cur_train_loss)
            train_acc_hist.append(cur_train_acc)
            val_loss_hist.append(cur_val_loss)
            val_acc_hist.append(cur_val_acc)


        #save final model
        
        # fast but also need to save out dimension of each layer or Net class(but my net class also need dimension of each layer to initialize)
        state = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': optimizer.state_dict()
                }
        torch.save(state, os.path.join(model_save_path,'last_model.pt'))
        
        
         ### graph train hist ###
        import matplotlib.pyplot as plt
        plt.style.use('ggplot')
    
        fig = plt.figure()
        plt.plot(train_loss_hist)
        plt.plot(val_loss_hist)
        plt.legend(['train loss','val loss'],loc='best')
        plt.savefig(os.path.join(model_save_path,'loss.jpg'))
        plt.close(fig)
        fig = plt.figure()
        plt.plot(train_acc_hist)
        plt.plot(val_acc_hist)
        plt.legend(['train acc','val acc'],loc='best')
        plt.savefig(os.path.join(model_save_path,'acc.jpg'))
        plt.close(fig)

        
        
    def predict_prob(self,df):
        
        #np_data = self.transformer.transform(df)
        tensor_data = torch.from_numpy(np.array(df)).to(self.device)
        

        with torch.no_grad():
            log_prob = F.log_softmax(self.model(tensor_data.float()))
            pred_prob = torch.exp(log_prob).data.cpu().numpy()
        
        return pred_prob
    
    
    def predict(self,df):
        pred_prob = self.predict_prob(df)
        pred_ind = np.argmax(pred_prob,axis=1)
        
        return pred_ind
    
    
    
    
    def train(self, optimizer, epoch):
        
        device = self.device
        train_loader = self.train_loader
        
        
        print('[epoch %d]train on %d data......'%(epoch,len(train_loader.dataset)))
        train_loss = 0
        correct = 0
        self.model.train()
        for batch_ind,(data,target) in enumerate(tqdm(train_loader)):
            data,target = data.to(device),target.to(device)
            optimizer.zero_grad()
            output = self.model(data.float())
            
            weights =  torch.tensor(self.class_weights).to(device)
            criterion = FocalLoss(weight=weights)
            #criterion = nn.CrossEntropyLoss(weight = weights)
            loss = criterion(output, target)
    
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
    
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
            #if batch_ind %30==0: #print during batch runing
            #    print('Train Epoch: %d [%d/%d](%.2f%%)\tLoss:%.6f' %(epoch, batch_ind*len(data),
            #          len(train_loader.dataset),100.*batch_ind/len(train_loader),loss.item() ))
    
        train_loss /= len(train_loader.dataset)
        acc = correct/len(train_loader.dataset)
    
        print('training set: average loss:%.4f, acc:%d/%d(%.3f%%)' %(train_loss,
              correct, len(train_loader.dataset), 100*acc))
    
        return train_loss, acc
    
    
    
    
    def val(self):
        
        model = self.model
        device = self.device
        val_loader = self.val_loader
        
        
        print('validation on %d data......'%len(val_loader.dataset))
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad(): #temporarily set all the requires_grad flag to false
            for data,target in val_loader:
                data,target = data.to(device),target.to(device)
                output = model(data.float())
                
                weights =  torch.tensor(self.class_weights).to(device)
                criterion = FocalLoss(weight=weights)
                #criterion = nn.CrossEntropyLoss(weight=weights)
                val_loss += criterion(output, target).item() #sum up batch loss
    
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
            val_loss /= len(val_loader.dataset) #avg of sum of batch loss
    
        print('Val set:Average loss:%.4f, acc:%d/%d(%.3f%%)' %(val_loss,
              correct, len(val_loader.dataset), 100.*correct/len(val_loader.dataset)))
    
        return val_loss, correct/len(val_loader.dataset)
            
    

# since i don't want boosting user need to install pytorch so i didn't put this loss fn in loss.py
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.nll_loss = nn.NLLLoss(weight)
        
    def forward(self, inputs, targets):      
        return self.nll_loss( (1-F.softmax(inputs,1))**self.gamma * F.log_softmax(inputs,1),targets )
    
    
