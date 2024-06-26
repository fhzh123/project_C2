# Import modules
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
# Import PyTorch
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# Import word importance realted modules
from lime.lime_text import LimeTextExplainer
from captum.attr import LayerIntegratedGradients, TokenReferenceBase
# Import custom modules
from utils.data_utils import data_load
from utils.tqdm_utils import TqdmLoggingHandler, write_log

def important_word_extract(args):
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
    if args.total_split_num == 1:
        pass
    else:
        if args.total_split_num < args.split_num:
            raise Exception("'split_num' is greater than 'total_split_num'. Check the number")
        else:
            split_ix = [x.tolist() for x in np.array_split(range(len(total_src_list['train'])), args.total_split_num)]

            total_src_list['train'] = total_src_list['train'][split_ix[args.split_num]]

    # Data dictionary setting
    data_ = dict()
    for i in range(len(total_src_list['train'])):
        data_[i] = dict()

    if args.data_name == 'IMDB':
        model_name_list = ['lvwerra/bert-imdb', 'fabriceyhc/bert-base-uncased-imdb', 'aychang/roberta-base-imdb', 'lvwerra/distilbert-imdb', 'JiaqiLee/imdb-finetuned-bert-base-uncased']
    elif args.data_name == 'SNLI':
        model_name_list = ['Alireza1044/albert-base-v2-mnli', "cross-encoder/nli-distilroberta-base", 'cross-encoder/nli-deberta-v3-base', 'sileod/deberta-v3-base-tasksource-nli', 'pepa/roberta-small-snli']
    elif args.data_name == 'RTE':
        model_name_list = ['textattack/roberta-base-RTE', 'gchhablani/bert-base-cased-finetuned-rte', 'yoshitomo-matsubara/bert-large-uncased-rte', 'textattack/distilbert-base-uncased-RTE', 'JeremiahZ/roberta-base-rte']
    elif args.data_name == 'SST2':
        model_name_list = ['Duopero/distilbert-base-uncased-finetuned-2-sst2', 'syedkhalid076/RoBERTa-Sentimental-Analysis-Model', 'Kai1014/distilbert-finetuned', 'distilbert/distilbert-base-uncased-finetuned-sst-2-english', 'echarlaix/bert-base-uncased-sst2-acc91.1-d37-hybrid']
    elif args.data_name == 'TREC6':
        model_name_list = ['aychang/bert-base-cased-trec-coarse', 'carrassi-ni/bert-base-trec-question-classification', 'aychang/distilbert-base-cased-trec-coarse']
    elif args.data_name == 'rotten':
        model_name_list = ['RJZauner/distilbert_rotten_tomatoes_sentiment_classifier', 'xianzhew/distilbert-base-uncased_rotten_tomatoes', 'jjglilleberg/distilbert-base-uncased-finetuned-rotten-tomatoes', 'Hazqeel/electra-small-finetuned-sst2-rotten_tomatoes-distilled', 'pig4431/rtm_ELECTRA_5E']
    elif args.data_name == 'yahoo_topic':
        model_name_list = []

    for model_name in model_name_list:

        write_log(logger, f"{model_name} model start...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        model.to(device)

        if args.data_name == 'IMDB' or args.data_name == 'SST2':
            class_names = ['negative', 'positive']

        def predict(input_ids, attention_mask, token_type_ids=None):
            if token_type_ids is None:
                output = model(input_ids=input_ids, attention_mask=attention_mask)[0]
            else:
                output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
            output = torch.softmax(output, dim=1)
            return output

        def predictor(texts):
            encoded_dict = tokenizer(texts, return_tensors="pt", padding=True)
            input_ids = encoded_dict['input_ids'].to(device)
            attention_mask = encoded_dict['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probas = F.softmax(outputs.logits, dim=1).cpu().detach().numpy()
            return probas

        if args.word_importance_method == 'Integrated_Gradients':
            lig = LayerIntegratedGradients(predict, model.get_input_embeddings())

            token_reference = TokenReferenceBase(reference_token_idx=tokenizer.pad_token_id)
            reference_indices = token_reference.generate_reference(args.src_max_len, device=device).unsqueeze(0)
        elif args.word_importance_method == 'Lime':
            explainer = LimeTextExplainer(class_names=class_names)

        for i, text in enumerate(tqdm(total_src_list['train'])):

            model.zero_grad()

            important_word = list()
            word_ix_list = list()

            if args.word_importance_method == 'Integrated_Gradients':
                if args.data_name == 'IMDB' or args.data_name == 'SST2':
                    encoded_dict = tokenizer(text, 
                                                max_length=args.src_max_len,
                                                padding='max_length',
                                                truncation=True,
                                                return_tensors='pt'
                                                )
                elif args.data_name == 'SNLI' or args.data_name == 'RTE':
                    encoded_dict = tokenizer(text[0], text[1], 
                                                max_length=args.src_max_len,
                                                padding='max_length',
                                                truncation=True,
                                                return_tensors='pt'
                                                )
                input_ids = encoded_dict['input_ids'].to(device)
                attention_mask = encoded_dict['attention_mask'].to(device)
                try:
                    token_type_ids = encoded_dict['token_type_ids'].to(device)
                except KeyError:
                    token_type_ids = None

                word_ids_list = np.array(encoded_dict.word_ids())
                new_word_ids_list = list()
                none_count = 0

                for i, w in enumerate(word_ids_list):
                    if i == 0 and w == None:
                        new_word_ids_list.append(none_count)
                        none_count += 1
                    elif w == None:
                        new_word_ids_list.append(new_word_ids_list[-1]+none_count)
                    else:
                        new_word_ids_list.append(w + none_count)

                with torch.no_grad():
                    pred = predict(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                    pred_ind = pred[0].argmax(dim=0).item()

                # attributions_ig, delta = lig.attribute(input_ids, reference_indices,
                #                                     additional_forward_args=(attention_mask, token_type_ids),
                #                                     n_steps=args.n_steps, return_convergence_delta=True, internal_batch_size=args.batch_size, target=pred_ind)
                attributions_ig = lig.attribute(input_ids, reference_indices,
                                                    additional_forward_args=(attention_mask, token_type_ids),
                                                    n_steps=args.n_steps, return_convergence_delta=False, internal_batch_size=args.batch_size, target=pred_ind)


                attributions_ = attributions_ig.sum(dim=2).squeeze(0)
                attributions_ = attributions_ / torch.norm(attributions_)

                for ix in attributions_.topk(args.word_importance_topk)[1]:

                    if ix == 0 or ix == args.src_max_len-1:
                        word_ = tokenizer.decode(input_ids[0][ix])
                    elif new_word_ids_list[ix-1] == new_word_ids_list[ix] or new_word_ids_list[ix+1] == new_word_ids_list[ix]:
                        total_word_index_list = np.where(np.array(new_word_ids_list) == new_word_ids_list[ix])[0]
                        word_ = tokenizer.decode(input_ids[0][total_word_index_list])
                    else:
                        word_ = tokenizer.decode(input_ids[0][ix])
                    
                    important_word.append(word_)

                total_results = [(a,b.item(),c) for a,b,c in zip(important_word, attributions_.topk(args.word_importance_topk)[1], attributions_[attributions_.topk(args.word_importance_topk)[1]].tolist())]

            elif args.word_importance_method == 'Lime':
                exp = explainer.explain_instance(text, predictor, num_features=args.word_importance_topk, num_samples=500)
                total_results = exp.as_list()

                # important_word = [x[0] for x in total_results]
                # score_list_ = [x[1] for x in total_results]
            if args.data_name == 'SNLI' or args.data_name == 'RTE':
                data_[i]['text1'] = text[0]
                data_[i]['text2'] = text[1]
            else:
                data_[i]['text'] = text
            data_[i][model_name] = total_results

    save_path = os.path.join(args.preprocess_path, f'important_word_{args.word_importance_method}_{args.data_name}_{args.word_importance_topk}_{args.split_num}_{args.total_split_num}.json')
    with open(save_path, 'w') as f:
        json.dump(data_, f)