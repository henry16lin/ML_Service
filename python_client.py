import requests
import json
import numpy as np
import time
#import pandas as pd
import argparse
import threading



class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def sent_request(args):
            
    url = server_url + '/' + args.model
    begin = time.time()
    
    #json_data = json.dumps(data.to_dict('list'),cls=NumpyEncoder)
    #print('sent request with json data:')
    #print(json_data)
    
    response = requests.post(url, json=json_data)
    response_dict = json.loads(response.text)
    
    end = time.time()
    print('elapse %.4f s'%(end-begin))
    
    print(response_dict['pred_prob'])
    #print('receive the model result:')
    #print(response_dict)

    shap_values_np = np.array(response_dict['shap_values'][0])

    postive_feature_ind = list(np.where(shap_values_np>0)[0])
    negative_feature_ind = list(np.where(shap_values_np<0)[0])

    feature_names_np = np.array(response_dict['feature_names'])
    positive_feature_np = feature_names_np[postive_feature_ind]
    negative_feature_np = feature_names_np[negative_feature_ind]

    positive_shap_np = shap_values_np[postive_feature_ind]
    negative_shap_np = shap_values_np[negative_feature_ind]

    positive_sort_index = np.argsort(-positive_shap_np)
    negative_sort_index = np.argsort(negative_shap_np)
    
    print('positive factors:')
    print( positive_feature_np[positive_sort_index] )
    print(positive_shap_np[positive_sort_index])

    print('\nnegative factors:')
    print( negative_feature_np[negative_sort_index])
    print(negative_shap_np[negative_sort_index])

    
    

def multi_send(args, thread, times):
    
    for i in range(times):
        print('thread: %d' %thread)
        sent_request(args)
    
    #time.sleep(5)
    
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'a python client test')
    parser.add_argument('--data_dir', type=str, help='path to data')
    parser.add_argument('--port', default='5566', help='port')
    parser.add_argument('--model', type=str, default='XGB', help='API name')
    parser.add_argument('--thread', type=int, default=1, help='how many thread used to send request')
    parser.add_argument('--times', type=int, default=1, help='request times in each thread')
    args = parser.parse_args()
    
    server_url = 'http://127.0.0.1:' + args.port

    #data = pd.read_csv(args.data_dir)

    with open(args.data_dir, 'r') as file:
        json_data = json.load(file)

    
    if args.thread == 1: 
        sent_request(args)
        
    elif(args.thread >1):
        print('create multi-thread to send request...')
        
        thread_start = time.time()
        for i in range(args.thread):
            t = threading.Thread(target = multi_send, args=(args, i, args.times) )
            t.start()
        
        t.join()
        time_cost = time.time()-thread_start
        time.sleep(1)
        print('finish all request, total elapse %.4f s'%time_cost)
        





