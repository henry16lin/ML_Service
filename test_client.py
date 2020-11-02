import requests
import json
import numpy as np
import time
import pandas as pd
#import argparse
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
    json_data = data
    

    response = requests.post(url, json=json_data)
    response_dict = json.loads(response.text)
    
    end = time.time()
    print(response_dict['pred_prob'])
    #print('receive the model result:')
    #print(response_dict)
    
    print('elapse %.4f s'%(end-begin))
    
    
def multi_send(args, thread, times):
    
    for i in range(times):
        print('thread: %d' %thread)
        sent_request(args)
    
    #time.sleep(5)
    
    
    

if __name__ == '__main__':
    
    class my_args():
        def __init__(self):
            self.data_dir = None
            self.model = None
            self.request = 1
            self.times = 1

        
    args = my_args()
    print('\n')
    server_url = input('server url (default localhost): ')
    if server_url == "":
        server_url = 'http://127.0.0.1:5566'
        print('use url:%s'%server_url)
        
    args.model = input('API: ')
    args.data_dir = input('data dir: ')
    
    
    #data = pd.read_csv(args.data_dir)
    with open(args.data_dir, 'r') as json_file:
        data = json.load(json_file)

    
    while True:
        args.request = int(input('client count: '))
        args.times = int(input('request count in each client:'))
        

        if args.request == 1: 
            sent_request(args)
            
        elif(args.request >1):
            print('create multi-thread to send request...')
            
            thread_start = time.time()
            for i in range(args.request):
                t = threading.Thread(target = multi_send, args=(args, i, args.times) )
                t.start()
            
            t.join()
            time_cost = time.time()-thread_start
            time.sleep(1)
            print('finish all request, total elapse %.4f s\n'%time_cost)







