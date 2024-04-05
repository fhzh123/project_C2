import torch
from torch.utils.data.dataset import Dataset

class Seq2LabelDataset(Dataset):
    def __init__(self, src_tokenizer, src_list: list = list(), trg_list: list = None, 
                 min_len: int = 10, src_max_len: int = 300):

        self.src_tensor_list = list()
        self.trg_tensor_list = list()
        
        self.src_tokenizer = src_tokenizer

        self.min_len = min_len
        self.src_max_len = src_max_len

        assert len(src_list) == len(trg_list)
        for src, trg in zip(src_list, trg_list):
            self.src_tensor_list.append(src)
            self.trg_tensor_list.append(trg)
        
        self.num_data = len(self.src_tensor_list)

    def __getitem__(self, index):
        if len(self.src_tensor_list[index]) == 2:
            src_encoded_dict = \
                self.src_tokenizer(
                    self.src_tensor_list[index][0], self.src_tensor_list[index][1],
                    max_length=self.src_max_len,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
        else:
            src_encoded_dict = \
                self.src_tokenizer(
                    self.src_tensor_list[index],
                    max_length=self.src_max_len,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
        # src_input_ids = src_encoded_dict['input_ids'].squeeze(0)
        # src_attention_mask = src_encoded_dict['attention_mask'].squeeze(0)

        trg_label = self.trg_tensor_list[index]

        return (src_encoded_dict, trg_label)

    def __len__(self):
        return self.num_data
    
class MaskingCustomDataset(Dataset):
    def __init__(self, tokenizer, src_list: list = list(), trg_list: list = None, 
                 masking_ix: int = 50264, min_len: int = 4, src_max_len: int = 300):

        self.tokenizer = tokenizer
        self.src_tensor_list = list()
        self.trg_tensor_list = list()

        self.min_len = min_len
        self.src_max_len = src_max_len

        for src in src_list:
            if min_len <= len(src):
                self.src_tensor_list.append(src)

        self.trg_tensor_list = trg_list

        self.masking_ix = masking_ix
        self.num_data = len(self.src_tensor_list)

    def __getitem__(self, index):
        encoded_dict = \
        self.tokenizer(
            self.src_tensor_list[index],
            max_length=self.src_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded_dict['input_ids'].squeeze(0)
        attention_mask = encoded_dict['attention_mask'].squeeze(0)

        mask_input_ids = torch.tensor(input_ids)

        # list_input = encoded_dict['input_ids'][0].cpu().tolist()
        # eos_ix = list_input.index(2)
        # sample_ix = sample(list(range(1, eos_ix)), int(eos_ix*0.2))
        # pre_input = [x if i not in sample_ix else 50264 for i,x in enumerate(list_input)]
        # mask_input_ids = torch.LongTensor(pre_input)#.unsqueeze(0)

        where_bool = mask_input_ids==2
        indices = where_bool.nonzero()
        sample_ix_list = torch.arange(1, indices[0][0])
        perm = torch.randperm(sample_ix_list.size(0))
        idx = perm[:int(indices * 0.2)]
        sample_ix = sample_ix_list[idx]
        mask_input_ids[sample_ix] = self.masking_ix

        trg_tensor = torch.tensor(self.trg_tensor_list[index], dtype=torch.long)
        return (input_ids, mask_input_ids, attention_mask, trg_tensor)

    def __len__(self):
        return self.num_data
    
class Seq2SeqDataset(Dataset):
    def __init__(self, src_tokenizer, trg_tokenizer, src_list: list = list(), trg_list: list = None, 
                 min_len: int = 10, src_max_len: int = 300, trg_max_len: int = 300):

        self.src_tensor_list = list()
        self.trg_tensor_list = list()
        
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer

        self.min_len = min_len
        self.src_max_len = src_max_len
        self.trg_max_len = trg_max_len

        assert len(src_list) == len(trg_list)
        for src, trg in zip(src_list, trg_list):
            if min_len <= len(src):
                self.src_tensor_list.append(src)
                self.trg_tensor_list.append(trg)
        
        self.num_data = len(self.src_tensor_list)

    def __getitem__(self, index):
        src_encoded_dict = \
        self.src_tokenizer(
            self.src_tensor_list[index],
            max_length=self.src_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        src_input_ids = src_encoded_dict['input_ids'].squeeze(0)
        src_attention_mask = src_encoded_dict['attention_mask'].squeeze(0)

        trg_encoded_dict = \
        self.trg_tokenizer(
            self.trg_tensor_list[index],
            max_length=self.trg_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        trg_input_ids = trg_encoded_dict['input_ids'].squeeze(0)
        trg_attention_mask = trg_encoded_dict['attention_mask'].squeeze(0)

        return (src_input_ids, src_attention_mask), (trg_input_ids, trg_attention_mask)

    def __len__(self):
        return self.num_data