import os
import numpy as np
from datasets import load_dataset

def data_split_index(data_len: int, split_ratio: float = 0.1):

    split_num = int(data_len * split_ratio)

    split_index = np.random.choice(data_len, split_num, replace=False)
    origin_index = list(set(range(data_len)) - set(split_index))

    return origin_index, split_index

def data_load(data_path:str = None, data_name:str = None):

    total_src_list, total_trg_list = dict(), dict()

    if data_name == 'WMT2016_Multimodal':

        data_path = os.path.join(data_path,'WMT/2016/multi_modal')

        # 1) Train data load
        with open(os.path.join(data_path, 'train.de'), 'r') as f:
            total_src_list['train'] = np.array([x.replace('\n', '') for x in f.readlines()])
        with open(os.path.join(data_path, 'train.en'), 'r') as f:
            total_trg_list['train'] = np.array([x.replace('\n', '') for x in f.readlines()])

        # 2) Valid data load
        with open(os.path.join(data_path, 'val.de'), 'r') as f:
            total_src_list['valid'] = np.array([x.replace('\n', '') for x in f.readlines()])
        with open(os.path.join(data_path, 'val.en'), 'r') as f:
            total_trg_list['valid'] = np.array([x.replace('\n', '') for x in f.readlines()])

        # 3) Test data load
        with open(os.path.join(data_path, 'test.de'), 'r') as f:
            total_src_list['test'] = np.array([x.replace('\n', '') for x in f.readlines()])
        with open(os.path.join(data_path, 'test.en'), 'r') as f:
            total_trg_list['test'] = np.array([x.replace('\n', '') for x in f.readlines()])

    if data_name == 'IMDB':

        dataset = load_dataset("imdb")

        origin_index, split_index = data_split_index(data_len=len(dataset['test']['text']), split_ratio=0.5)

        # 1) Train data load
        total_src_list['train'] = np.array(dataset['train']['text'])
        total_trg_list['train'] = np.array(dataset['train']['label'])

        # 2) Valid data load
        total_src_list['valid'] = np.array(dataset['test']['text'])[origin_index]
        total_trg_list['valid'] = np.array(dataset['test']['label'])[origin_index]

        # 3) Test data load
        total_src_list['test'] = np.array(dataset['test']['text'])[split_index]
        total_trg_list['test'] = np.array(dataset['test']['label'])[split_index]

    if data_name == 'SST2':

        dataset = load_dataset("sst2")

        # 1) Train data load
        total_src_list['train'] = np.array(dataset['train']['sentence'])
        total_trg_list['train'] = np.array(dataset['train']['label'])

        # 2) Valid data load
        total_src_list['valid'] = np.array(dataset['validation']['sentence'])
        total_trg_list['valid'] = np.array(dataset['validation']['label'])

        # 3) Test data load
        total_src_list['test'] = np.array(dataset['test']['sentence'])
        total_trg_list['test'] = np.array(dataset['test']['label'])

    if data_name == 'tweet_emoji':

        dataset = load_dataset("tweet_eval", "emoji")

    if data_name == 'SNLI':

        dataset = load_dataset("snli")

        hypothesis_train = dataset['train']['hypothesis']
        premise_train = dataset['train']['premise']
        total_src_list['train'] = np.array([(h, p) for h, p in zip(hypothesis_train, premise_train)])
        total_trg_list['train'] = np.array(dataset['train']['label'])

        hypothesis_valid = dataset['validation']['hypothesis']
        premise_valid = dataset['validation']['premise']
        total_src_list['valid'] = np.array([(h, p) for h, p in zip(hypothesis_valid, premise_valid)])
        total_trg_list['valid'] = np.array(dataset['validation']['label'])

        hypothesis_test = dataset['test']['hypothesis']
        premise_test = dataset['test']['premise']
        total_src_list['test'] = np.array([(h, p) for h, p in zip(hypothesis_test, premise_test)])
        total_trg_list['test'] = np.array(dataset['test']['label'])

    if data_name == 'RTE':

        dataset = load_dataset("nyu-mll/glue", "rte")

        hypothesis_train = dataset['train']['sentence1']
        premise_train = dataset['train']['sentence2']
        total_src_list['train'] = np.array([(h, p) for h, p in zip(hypothesis_train, premise_train)])
        total_trg_list['train'] = np.array(dataset['train']['label'])

        hypothesis_valid = dataset['validation']['sentence1']
        premise_valid = dataset['validation']['sentence2']
        total_src_list['valid'] = np.array([(h, p) for h, p in zip(hypothesis_valid, premise_valid)])
        total_trg_list['valid'] = np.array(dataset['validation']['label'])

        hypothesis_test = dataset['test']['sentence1']
        premise_test = dataset['test']['sentence2']
        total_src_list['test'] = np.array([(h, p) for h, p in zip(hypothesis_test, premise_test)])
        total_trg_list['test'] = np.array(dataset['test']['label'])

    return total_src_list, total_trg_list