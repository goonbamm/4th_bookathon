import os
import sys
import json
import torch
import random

import numpy as np
import pandas as pd

from filtering import *


class Logger(object):
    def __init__(self, CONFIG):
        self.terminal = sys.stdout
        self.log = open(os.path.join(CONFIG['result_path'], 'console.log', ), 'w', encoding='utf-8')            
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass    


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_config(file_path=''):
    with open(file_path, 'r', encoding='utf-8') as f:
        CONFIG = json.load(f)
    
    return CONFIG


def mkdir_result_dir(CONFIG):
    file_number = 0
    
    if not os.path.exists(CONFIG['result_path']):
        os.mkdir(CONFIG['result_path'])
    
    else:
        file_number = len(os.listdir(CONFIG['result_path']))
        
    CONFIG['result_path'] = os.path.join(CONFIG['result_path'], str(file_number))
    os.mkdir(os.path.join(CONFIG['result_path']))
    
    with open(os.path.join(CONFIG['result_path'], 'hyperparameter.json'), 'w', encoding='utf-8') as f:
        json.dump(CONFIG, f, indent=4)
    

def get_all_datapaths(CONFIG):
    csv_list = list()
    data_path = CONFIG['data_path']

    if isinstance(data_path, list):
        for dp in data_path:
            csv_list += os.listdir(data_path)
    
    else: # str
        csv_list += os.listdir(data_path)
    
    return csv_list


def get_all_datasets(CONFIG, csv_list):
    data_path = CONFIG['data_path']
    df_dataset = pd.DataFrame()

    for csv_path in csv_list:
        try:
            tmp_csv = pd.read_csv(os.path.join(data_path, csv_path), encoding='utf-8', engine='python')

        except:
            continue

        df_dataset = pd.concat([df_dataset, tmp_csv], ignore_index=True)

    df_dataset = filtering(df_dataset)
    
    if CONFIG['data_type'] == 'sentence':
        df_dataset = doc2sent(df_dataset)

    df_dataset['content'] = df_dataset['content'].apply(clear_text)
        
    return df_dataset
