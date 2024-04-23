import os
import json
import logging
from tqm import tqdm
from random import random

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, PrefixTuningConfig, TaskType, PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, default_data_collator, get_linear_schedule_with_warmup
# Import custom modules
from project_C2.model.dataset import MaskingCustomDataset
from utils.data_utils import data_load
from utils.tqdm_utils import TqdmLoggingHandler, write_log

def mask_infilling(args):
    if args.cuda:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            raise Exception('Cuda is not available. Check again')
    else:
        device = torch.device('cpu')

    #===================================#
    #==============Logging==============#
    #===================================#

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    write_log(logger, 'Start extracting!')

    #===================================#
    #=============Data Load=============#
    #===================================#

    write_log(logger, "Load data...")

    total_src_list, total_trg_list = data_load(data_path=args.data_path, data_name=args.data_name)

    # Data split
    src_train_list, trg_train_list = list(), list()
    src_valid_list, trg_valid_list = list(), list()
    for label in set(total_trg_list['train']):
        src_train_list.append([src for src, trg in zip(total_src_list['train'], total_trg_list['train']) if trg == label])
        trg_train_list.append([trg for _, trg in zip(total_src_list['train'], total_trg_list['train']) if trg == label])
        src_valid_list.append([src for src, trg in zip(total_src_list['valid'], total_trg_list['valid']) if trg == label])
        trg_valid_list.append([trg for _, trg in zip(total_src_list['valid'], total_trg_list['valid']) if trg == label])

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)

        dataset_dict = {
            'train': MaskingCustomDataset(tokenizer=tokenizer, src_list=total_src_list['train'], trg_list=total_trg_list['train'],
                                        masking_ix=tokenizer.mask_token_id, min_len=args.min_len, src_max_len=args.src_max_len),
            'valid': MaskingCustomDataset(tokenizer=tokenizer, src_list=total_src_list['valid'], trg_list=total_trg_list['valid'],
                                        masking_ix=tokenizer.mask_token_id, min_len=args.min_len, src_max_len=args.src_max_len),
        }
        dataloader_dict = {
            'train': DataLoader(dataset_dict['train'], drop_last=True,
                                batch_size=args.batch_size, shuffle=True,
                                pin_memory=True, num_workers=args.num_workers),
            'valid': DataLoader(dataset_dict['valid'], drop_last=False,
                                batch_size=args.batch_size, shuffle=True, 
                                pin_memory=True, num_workers=args.num_workers)
        }

        peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, num_virtual_tokens=20)

        model = get_peft_model(model, peft_config)
        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(dataloader_dict['train']) * args.num_epochs),
        )

        best_loss = 1e+10

        for epoch in range(args.num_epochs):
            model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(dataloader_dict['train'])):
                input_ids = batch[0].to(device)
                mask_input_ids = batch[1].to(device)
                attention_mask = batch[2].to(device)
                label = batch[3].to(device)
                outputs = model(input_ids=mask_input_ids, attention_mask=attention_mask, labels=input_ids, task_ids=label)
                loss = outputs.loss
                total_loss += loss.detach().float()
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            model.eval()
            eval_loss = 0

            for step, batch in enumerate(tqdm(dataloader_dict['valid'])):
                input_ids = batch[0].to(device)
                mask_input_ids = batch[1].to(device)
                attention_mask = batch[2].to(device)
                label = batch[3].to(device)
                with torch.no_grad():
                    outputs = model(input_ids=mask_input_ids, attention_mask=attention_mask, labels=input_ids, task_ids=label)
                loss = outputs.loss
                eval_loss += loss.detach().float()

            eval_epoch_loss = eval_loss / len(dataloader_dict['valid'])
            eval_ppl = torch.exp(eval_epoch_loss)
            train_epoch_loss = total_loss / len(dataloader_dict['train'])
            train_ppl = torch.exp(train_epoch_loss)
            print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

            if eval_epoch_loss <= best_loss:
                write_log(logger, 'Checkpoint saving...')
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, f'checkpoint_{args.data_name}_{args.model_name}_label_{label}.pt')
                best_loss = eval_epoch_loss
                best_epoch = epoch