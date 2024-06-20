import jsonlines 
import os
from data_util import convert_to_span, read_jsonl_file
import torch 
import json 
import pandas as pd 
import json

tag2id = {"O": 0, "B-SPAN": 1, "I-SPAN": 2}


def shift_label(label):
    # If the label is B-SPAN make it I-SPAN.
    if label == 1:
        label == 2
    return label

    

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id is None:
            new_labels.append(-100)
        elif word_id != current_word:
            # Start of a new word!
            current_word = word_id
            
            new_labels.append(tag2id[labels[word_id]])
        else:
            new_labels.append(shift_label(tag2id[labels[word_id]]))

    return new_labels

def tokenize_and_align_labels_2(tokens, tags ,tokenizer,max_length=128):
    tokenized_inputs = tokenizer(tokens, truncation=True, is_split_into_words=True, return_tensors="pt",padding='max_length', max_length=max_length)
    new_labels = []
    for i, labels in enumerate(tags):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs   

def convert_to_bio(dataset):
    sentences = []
    spans = []
    for sample in dataset:
        if not sample.get("context"):
            continue
        context = sample['context']
        target_spans = sample['target_spans']
        sentences.append(context)
        spans.append(target_spans)
    all_tokens = []
    all_tags = []
    for sentence, span_list in zip(sentences, spans):
        tokens = sentence.split()
        tags = ['O'] * len(tokens)  # Initialize tags as 'Outside'
        token_start = 0
        token_end = 0
      

        for idx, token in enumerate(tokens):
            token_start = sentence.find(token, token_end)
            token_end = token_start + len(token)
            
            for start, end in span_list:

                
                if token_start == start:  # Token is at the beginning of a span
                    tags[idx] = 'B-SPAN'
                elif start < token_start < end:  # Token is inside a span
                    tags[idx] = 'I-SPAN'
        

        all_tokens.append(tokens)
        all_tags.append(tags)
        
       

    return all_tokens, all_tags

def convert_to_bio_sent(sample):
    context = sample['context']
    target_spans = sample['target_spans']
    tokens = context.split()
    tags = ['O'] * len(tokens)  # Initialize tags as 'Outside'
    token_start = 0
    token_end = 0


    for idx, token in enumerate(tokens):
        token_start = context.find(token, token_end)
        token_end = token_start + len(token)
        
        for start, end in target_spans:

            
            if token_start == start:  # Token is at the beginning of a span
                tags[idx] = 'B-SPAN'
            elif start < token_start < end:  # Token is inside a span
                tags[idx] = 'I-SPAN'
    

    return list(zip(tokens, tags))




    


class TargetSpanDataset(torch.utils.data.Dataset):
    def __init__(self, dataset,tokenizer,max_length = 128):
        self.tokenizer = tokenizer
        self.dataset = dataset
      
        self.max_length = max_length
        #first convert teh dataset to bio tags
        tokens,tags = convert_to_bio(self.dataset)
        
     
        #encode the data
        self.tokenized_input = tokenize_and_align_labels_2(tokens,tags,self.tokenizer,max_length=self.max_length)

        

    def __getitem__(self, idx):
        return {
        'input_ids': self.tokenized_input['input_ids'][idx],
        'labels': self.tokenized_input['labels'][idx],
        'context': self.dataset[idx]['context'],
        'target_spans': self.dataset[idx]['target_spans'],
        'targets': self.dataset[idx]['targets'],
    }

    def __len__(self):
        return len(self.dataset)











    