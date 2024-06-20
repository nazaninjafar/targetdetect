from transformers import RobertaTokenizerFast, BertTokenizerFast
from transformers import (AutoModelForTokenClassification,
        TrainingArguments,
        Trainer,
 )
from data_loader import TargetSpanDataset
from data_util import read_jsonl_file,convert_predicted_to_spans,convert_tensor_to_spans
from transformers import DataCollatorForTokenClassification
import evaluate
import numpy as np 
import torch
from data_loader import convert_to_bio_sent
import argparse
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
import json
import jsonlines
import os 


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bert-base-uncased', help='encoder name')
    parser.add_argument('--dataset', type=str, default='implicit-hate-corpus', help='Dataset name')
    parser.add_argument('--test_dataset', type=str, default='implicit-hate-corpus', help='Dataset name')
    parser.add_argument('--fewshot', type=bool, default=False, help='Train full model')
    return parser.parse_args()



seqeval = evaluate.load("seqeval")
label_list = ["O", "B-SPAN", "I-SPAN"]
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

id2label = {
    0: "O", 1: "B-SPAN", 2: "I-SPAN"
   
}
label2id = {
    "O": 0, "B-SPAN": 1, "I-SPAN": 2
}

args = args()
model_name = args.model

model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=3, id2label=id2label, label2id=label2id)
if model_name == 'roberta-large':
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name,add_prefix_space=True)
else:
    tokenizer = BertTokenizerFast.from_pretrained(model_name,add_prefix_space=True)
#replace special token ids with -100 

#print special token ids

dataset = args.dataset
test_data_name = args.test_dataset
all_datasets = ['implicit-hate-corpus','dynahate','sbic']
train_dataset = []
dev_dataset = []
if dataset =='all':
    for d in all_datasets:
        train_dataset += read_jsonl_file(f'data/targeted/{d}/gpt3.5_train_spanned.jsonl')
        dev_dataset += read_jsonl_file(f'data/targeted/{d}/gpt3.5_dev_spanned.jsonl')

    
else:
    if dataset =='plead':
        train_dataset = read_jsonl_file(f'data/targeted/{dataset}/train_spanned_hate.jsonl')
        dev_dataset = read_jsonl_file(f'data/targeted/{dataset}/dev_spanned_hate.jsonl')
       
    else:
        train_dataset = read_jsonl_file(f'data/targeted/{dataset}/gpt3.5_train_spanned.jsonl')
        if args.fewshot:
            #randomly sample 50 samples from the training set
            train_dataset = np.random.choice(train_dataset,50)
        dev_dataset = read_jsonl_file(f'data/targeted/{dataset}/gpt3.5_dev_spanned.jsonl')

if test_data_name =='plead':
    test_dataset = read_jsonl_file(f'data/targeted/{test_data_name}/test_spanned_hate.jsonl')
else:     
    test_dataset = read_jsonl_file(f'data/targeted/{test_data_name}/gpt3.5_test_spanned.jsonl')


# test_dataset = test_dataset[:10]
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

train_dataset = TargetSpanDataset(train_dataset,tokenizer)

dev_dataset = TargetSpanDataset(dev_dataset,tokenizer)

test_dataset = TargetSpanDataset(test_dataset,tokenizer)

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=5,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    load_best_model_at_end=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,

)

trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=dev_dataset,             # evaluation dataset
    compute_metrics=compute_metrics,
    data_collator=data_collator
)
trainer.train()

# # save the model
if args.fewshot:
    model.save_pretrained(f'./trained_mutas_{dataset}_{model_name}_fewshot')
else:
    model.save_pretrained(f'./trained_mutas_{dataset}_{model_name}')







def get_mismatched_tokens(predicted,true):
    mismatched = []
    for i,t in enumerate(true):
        if t != predicted[i]:
            mismatched.append((i,t,predicted[i]))
        
    return mismatched

def tensor_to_tags(tensor):
    tags = []
    for token_class in tensor:
        token_class = token_class.item()
        if token_class == 0 or token_class == -100:
            tags.append('O')
        elif token_class == 1:
            tags.append('B-SPAN')
        elif token_class == 2:
            tags.append('I-SPAN')
    return tags

def eval_model(model_id,dataset):
    model = AutoModelForTokenClassification.from_pretrained(model_id)
    model.eval()
    true_predictions = []
    true_labels = []
    missclassified_samples = []
    # Dictionary of special token IDs
    special_token_ids = {
        'bos_token': tokenizer.bos_token_id,
        'eos_token': tokenizer.eos_token_id,
        'pad_token': tokenizer.pad_token_id,
        # 'unk_token': tokenizer.unk_token_id,
        'sep_token': tokenizer.sep_token_id,
        'cls_token': tokenizer.cls_token_id,
        'mask_token': tokenizer.mask_token_id
    }
    
    # Remove None values (for tokenizers that don't use all special tokens)
    special_token_ids = {k: v for k, v in special_token_ids.items() if v is not None}

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for batch_idx,batch in enumerate(tqdm(data_loader)):
    
        # Get special token IDs


        # Remove None values (for tokenizers that don't use all special tokens)
        
        missclassified={}
        input_ids = batch['input_ids']
        
        true_label = batch['labels']

       
        with torch.no_grad():
            logits = model(input_ids).logits
        predictions = torch.argmax(logits, dim=2)
        
        #remove special tokens from predictions
        predictions = predictions[0]
      
        # predictions = [t for i,t in enumerate(predictions) if i not in special_token_indices]
        
        predicted_token_class = tensor_to_tags(predictions)
        # print("predicted_token_class",predicted_token_class)
        # true_label = [model.config.id2label[t.item()] for t in true_label if t.item() != -100]
        true_label = tensor_to_tags(true_label)
        # print("true_label",true_label)
        
        true_labels.append(true_label)
        # print(len(predicted_token_class))
        # print(len(true_label))

    
        assert len(predicted_token_class) == len(true_label)
        
        predicted_spans = convert_predicted_to_spans(predicted_token_class)

        true_spans = convert_predicted_to_spans(true_label)
        
        #if predicted spans are not equal to true spans, add to missclassified samples
        if not args.fewshot:
            if predicted_spans != true_spans:
                words = tokenizer.convert_ids_to_tokens(input_ids[0])
                

                missed_target_tokens = []
                missed_spans = []
                for span in true_spans:
                    if span not in predicted_spans:
                        missed_target_tokens.append(' '.join(words[span[0]:span[-1]+1]))
                        missed_spans.append(span)
                # missclassified['predicted_spans'] = predicted_spans
                missclassified['missed_target_tokens'] = missed_target_tokens
                missclassified["predicted_tokens"] = [' '.join(words[span[0]:span[-1]+1]) for span in predicted_spans]
                missclassified['true_tokens'] = [' '.join(words[span[0]:span[-1]+1]) for span in true_spans]
                missclassified['context'] = batch['context']
                missclassified['missed_spans'] = missed_spans
                missclassified['true_spans'] = true_spans
                missclassified['predicted_spans'] = predicted_spans
                # missclassified['target_spans'] = batch['target_spans']
                missclassified['targets'] = batch['targets']
        # print(missclassified)    
        # print(len(missclassified_samples))
        missclassified_samples.append(missclassified)        



        true_predictions.append(predicted_token_class)
        

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    },missclassified_samples







if args.fewshot:
    model_name = f'trained_mutas_{dataset}_{model_name}_fewshot'
else:
    model_name = f'trained_mutas_{dataset}_{model_name}'
results,misclassified_samples = eval_model(model_name,test_dataset)

#writing results to a file
results_dir = f'pred_results/{dataset}/{model_name}'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

if args.fewshot:
    results_file = f'{test_data_name}_fewshot.json'
else:
    results_file = f'{test_data_name}.json'

results_file = os.path.join(results_dir,results_file)
with open(results_file, 'w') as f:
    f.write(json.dumps(results))
    f.write('\n')

#wrie missclassified samples to a file
if not args.fewshot:
    missclassified_file = f'missclassified_samples_{dataset}_{model_name}_{test_data_name}.jsonl'
    with jsonlines.open(missclassified_file, 'w') as writer:
        writer.write_all(misclassified_samples)


