import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import jsonlines 
import nltk
import torch
from nltk.stem import WordNetLemmatizer
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np 
from transformers import AutoTokenizer
from collections import Counter
from difflib import SequenceMatcher

def longest_common_subsequence(seq1, seq2):
    """
    Calculate the length of the longest common subsequence between two sequences.
    """
    matcher = SequenceMatcher(None, seq1, seq2)
    lcs = sum(match.size for match in matcher.get_matching_blocks())
    return lcs

def lcs_agreement(annotations, text_length):
    """
    Calculate the inter-annotator agreement using the Longest Common Subsequence (LCS) method.
    annotations is a list of lists, where each inner list contains the spans annotated by one annotator.
    text_length is the length of the text being annotated.
    """
    # Convert span annotations to binary sequence representations
    binary_annotations = []
    for annotator_spans in annotations:
        binary_sequence = [0] * text_length
        for start, end in annotator_spans:
            for i in range(start, end):
                binary_sequence[i] = 1
        binary_annotations.append(binary_sequence)

    # Calculate pairwise LCS agreement
    pairwise_agreements = []
    for i in range(len(binary_annotations)):
        for j in range(i + 1, len(binary_annotations)):
            lcs_length = longest_common_subsequence(binary_annotations[i], binary_annotations[j])
            agreement = lcs_length / text_length
            pairwise_agreements.append(agreement)

    # Return the average agreement across all pairs
    return sum(pairwise_agreements) / len(pairwise_agreements) if pairwise_agreements else 0





device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
nltk.download('wordnet')

# Create a WordNet lemmatizer
lemmatizer = WordNetLemmatizer()


# p = inflect.engine()

def unify_targets(target):
    target = str(target)
    target = target.lower()
    target = target.replace('people','').strip()
    #replace the derivative words with the main word 
    target=lemmatize_main_word(target)
    target = plural_to_singular(target)
    return target

def plural_to_singular(word):
    # Check if the word is plural
    if p.singular_noun(word):
        return p.singular_noun(word)
    else:
        return word
# Function to lemmatize a phrase and keep the main word
def lemmatize_main_word(phrase):
    words = phrase.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    # Keep the first lemma as the main word
    main_word = lemmatized_words[0]
    return main_word



def pd_read_as_jsonl(df, output_path):
    with jsonlines.open(output_path, mode='w') as writer:
        for i in range(len(df)):
            writer.write(df.iloc[i].to_dict())



def read_jsonl(input_path):
    with jsonlines.open(input_path) as reader:
        data = [obj for obj in reader]
    return data


class TargetClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer,data_name='implicithate',fold='train'):
        if data_name=='implicithate':
            self.data = ToxiGenCorpus()
            self.dataset = self.data.get_jsonl_fold(fold)
            # self.dataset = self.dataset[:100]
            self.lookup_map,self.num_labels = self.data.get_lookup_map()
            print(self.num_labels)
            self.tokenizer = tokenizer

        
        
    def encode(self, text):
        encoded = self.tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
       
  
        return encoded

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        data_sample = self.dataset[index]
        text = data_sample['post']
        label = data_sample['target']
        label = self.lookup_map[label]
        encoding = self.encode(text)
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': torch.tensor(label, dtype=torch.long)
        }





class DynaHate:
    def __init__(self, data_path = 'data/dynahate/'):
        self.data_path = data_path
        
        




    def get_hates_only(self,dataset):
        hate_train_data = [data for data in dataset if data['hate'] == 1]
        return hate_train_data
    # def to_jsonl(self,dataset,output_path):
    #     jsonl_list = []
        
    #     for i in range(len(dataset)):
            
    #         jsonl_list.append({'post':dataset[i]['post'],
    #                            'hate':dataset[i]['hate'],
    #                            })

    #     with jsonlines.open(output_path,'w') as writer:
    #         for i in range(len(jsonl_list)):
    #             writer.write(jsonl_list[i])

    def read_jsonl(self,input_path):
        with jsonlines.open(input_path) as reader:
            data = [obj for obj in reader]
        return data

class ImplicitHateCorpus:
    def __init__(self, data_path = 'data/implicit-hate-corpus/'):
        self.data_path = data_path
        df = pd.read_csv(data_path+'implicit_hate_v1_stg3_posts.tsv',sep='\t',names = ['post','target','implied_statement'])
        train_df = df.sample(frac=0.8, random_state=42)
     
        dev_test_df = df.drop(train_df.index)
        dev_df = dev_test_df.sample(frac=0.5, random_state=42)
        test_df = dev_test_df.drop(dev_df.index)
       
        
        
    def get_targets(self,all_data_path='data/implicit-hate-corpus/implicit_hate_v1_stg3_posts.tsv'):
        all_data = pd.read_csv(all_data_path,sep='\t',names = ['post','target','implied_statement'])
        #remove nan 

        #remove the rows with nan targets
        all_data = all_data[~all_data['target'].isna()]
        targets = all_data['target'].tolist()
        return targets

    def get_jsonl_fold(self,fold):
        fold_data = read_jsonl(self.data_path + fold+'.jsonl')
        #remove the rows with nan targets
        fold_data = [data for data in fold_data if not data['target'] is None]
        return fold_data

    #i want you to write a code that would unify the data['target'] into few labels. For example a "german people" and "germans" should be the same label.
    def unify_targets(self,target):
        target = str(target)
        target = target.lower()
        target = target.replace('people','').strip()
        #replace the derivative words with the main word 
        target=lemmatize_main_word(target)
        target = plural_to_singular(target)
        return target

    def get_lookup_map(self):
        targets = self.get_targets()

        targets = [self.unify_targets(target) for target in targets]
        targets = list(set(targets))
        lookup_map = {}
        i = 0
        for target in targets:
            lookup_map[target] = i
            i+=1
        print(lookup_map)
        num_classes = len(targets)
        return lookup_map,num_classes

class SBIC:
    def __init__(self, data_path = 'data/SBIC/'):
        self.data_path = data_path
        #read with pandas 
       
        self.train_data = self.read_jsonl('data/sbic/train.jsonl')
        self.dev_data = self.read_jsonl('data/sbic/dev.jsonl')
        self.test_data = self.read_jsonl('data/sbic/test.jsonl')
        self.train_data = self.get_hates_only(self.train_data)
        self.dev_data = self.get_hates_only(self.dev_data)
        self.test_data = self.get_hates_only(self.test_data)
        with jsonlines.open('data/sbic/train_hate.jsonl','w') as writer:
            for i in range(len(self.train_data)):
                writer.write(self.train_data[i])
        with jsonlines.open('data/sbic/dev_hate.jsonl','w') as writer:
            for i in range(len(self.dev_data)):
                writer.write(self.dev_data[i])
        with jsonlines.open('data/sbic/test_hate.jsonl','w') as writer:
            for i in range(len(self.test_data)):
                writer.write(self.test_data[i])


    def to_jsonl(self,dataset,output_path):
        jsonl_list = []
        for i in range(len(dataset)):
            jsonl_list.append({'post':dataset.iloc[i]['post'],
                               'target':dataset.iloc[i]['targetMinority'],
                               'offensive':dataset.iloc[i]['offensiveYN'],
                               'implied_statement':dataset.iloc[i]['targetStereotype']})


        with jsonlines.open(output_path,'w') as writer:
            for i in range(len(jsonl_list)):
                writer.write(jsonl_list[i])

    def read_jsonl(self,input_path):
        with jsonlines.open(input_path) as reader:
            data = [obj for obj in reader]
        return data
    
    def get_hates_only(self,dataset):
        hate_dataset=[]
        unique_posts = []

        for data in dataset:
            if data['offensive']>0.0:
                if data['post'] in unique_posts:
                    continue
                unique_posts.append(data['post'])
                hate_dataset.append(data)
        return hate_dataset




    

class ToxiGenCorpus:
    def __init__(self):
        
        self.train_data = load_dataset("skg/toxigen-data", name="train", use_auth_token=True) # 250k training examples
        # self.to_jsonl(self.train_data['train'])
        
        
        self.train_data_human = load_dataset("skg/toxigen-data", name="annotated", use_auth_token=True) # Human study annotations
        # self.to_jsonl(self.train_data_human['train'])

        
    
    def to_jsonl(self,dataset):
        jsonl_list = []
        for i in range(len(dataset)):
            #if the toxicity_human is above 3 add it to the dataset 
            if dataset[i]['toxicity_human']>=3:
                jsonl_list.append({'post':dataset[i]['text'],'target':dataset[i]['target_group']})
        with jsonlines.open('data/toxigen-data/test_hate.jsonl','w') as writer:
            for i in range(len(jsonl_list)):
                writer.write(jsonl_list[i])


    def get_jsonl_fold(self,fold):
        fold_data = read_jsonl('data/toxigen-data/'+fold+'_hate.jsonl')
        return fold_data

    def get_lookup_map(self):
        targets = ['lgbtq', 'latino', 'muslim', 'chinese', 'mental_dis', 'physical_dis', 'asian', 'native_american', 'black', 'mexican', 'middle_east', 'women', 'jewish']
        lookup_map = {}
        i = 0
        for target in targets:
            lookup_map[target] = i
            i+=1

        num_classes = len(targets)
        return lookup_map,num_classes

    def unify_targets(self,target):
        target = str(target)
        target = target.lower()
        target = target.replace('people','').strip()
        #replace the derivative words with the main word 
        target=lemmatize_main_word(target)
        target = plural_to_singular(target)
        return target

class HateCheckCorpus:
    def __init__(self,test_data_path = 'data/hatecheck-data/test_suite_cases.csv', all_data_path = 'data/hatecheck-data/all_cases.csv'):

        self.test_data = pd.read_csv(test_data_path)
        # self.to_jsonl(self.test_data,'data/hatecheck-data/test.jsonl')
        # self.all = pd.read_csv(train_data_path)
        # #exclude test data from all data
        # self.train_data = self.all[~self.all['case_id'].isin(self.test_data['case_id'])]
        # self.dev_data = self.train_data.sample(frac=0.2, random_state=42)

        
    def to_jsonl(self,dataset,output_path):
        jsonl_list = []
        for i in range(len(dataset)):
            jsonl_list.append({'post':dataset.iloc[i]['test_case'],'target':dataset.iloc[i]['target_ident']})
        with jsonlines.open(output_path,'w') as writer:
            for i in range(len(jsonl_list)):
                writer.write(jsonl_list[i])

    def get_jsonl_fold(self,fold):
        fold_data = read_jsonl('data/hatecheck-data/'+fold+'.jsonl')
        # #remove nan targets
        # fold_data = [data for data in fold_data if not data['target'] is None]
        return fold_data


#datamodule for implicit hate corpus

#setup data module
class ClassificationDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.training_model = args.training_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.training_model ,cache_dir='/work/nazaninjafar_umass_edu/')
        

    def setup(self, stage=None):
        self.train_dataset = TargetClassificationDataset(self.tokenizer,fold='train')
        self.val_dataset = TargetClassificationDataset(self.tokenizer,fold='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.train_batch_size, shuffle=True, num_workers=1)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.args.eval_batch_size,shuffle=True, num_workers=1)
    
    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.args.eval_batch_size, num_workers=1)

def calculate_num_target_per_text(sample):
    #calculate the number of targets per text
    unique_targets = list(set(sample['targets']))
    num_targets = len(unique_targets)
    return num_targets

def calculate_target_token_length(sample):
    #calculate the length of the target
    target_lengths = []
    for target in sample['targets']:
        target_length = len(target.split())
        target_lengths.append(target_length)
    return target_lengths

def avg_target_token_length(dataset):
    target_lengths = []
    for sample in dataset:
        target_lens= calculate_target_token_length(sample)
        #avg target length per text
        avg_target_length = np.mean(target_lens)
        target_lengths.append(avg_target_length)
    avg_target_length = np.mean(target_lengths)
    stdev_target_length = np.std(target_lengths)
    return avg_target_length,stdev_target_length


def avg_num_targets_per_text(dataset):
    num_targets = []
    for sample in dataset:
        num_targets.append(calculate_num_target_per_text(sample))
    avg_num_targets = np.mean(num_targets)
    stdev_num_targets = np.std(num_targets)
    return avg_num_targets,stdev_num_targets





        


def calculate_overlap(span1, span2):
    """
    Calculate the number of overlapping characters between two spans.
    Each span is a tuple of (start, end).
    """
    return max(0, min(span1[1], span2[1]) - max(span1[0], span2[0]))

def calculate_agreement(annotations):
    """
    Calculate the agreement score between annotators with partial overlaps, including cases with empty annotations.
    annotations is a list where each element is an annotator's list of spans.
    """
    # Pairwise comparison of annotators
    pairwise_agreements = []
    for i in range(len(annotations)):
        for j in range(i+1, len(annotations)):
            annotator1 = annotations[i]
            annotator2 = annotations[j]
            total_overlap = 0
            total_chars = 0
            
            # Check each span in annotator1 against all spans in annotator2
            for span1 in annotator1:
                for span2 in annotator2:
                    total_overlap += calculate_overlap(span1, span2)
                
                # Add the number of characters in the current span to the total
                total_chars += span1[1] - span1[0]
            
            # Also add the characters from annotator2's spans that weren't considered
            total_chars += sum(span2[1] - span2[0] for span2 in annotator2)
            
            # Handle cases where one or both annotators provided no annotations
            if total_chars == 0:
                agreement_score = 1.0 if not annotator1 and not annotator2 else 0.0
            else:
                agreement_score = total_overlap / total_chars
            
            # Add the agreement score for this pair of annotators to the list
            pairwise_agreements.append(agreement_score)
    
    # Return the average agreement across all pairs
    return sum(pairwise_agreements) / len(pairwise_agreements) if pairwise_agreements else 0




def dice_coefficient(annotator1_spans, annotator2_spans):
    """
    Calculate the Dice Coefficient between two sets of spans from different annotators.
    The Dice Coefficient is 2 * |X ∩ Y| / (|X| + |Y|),
    where X and Y are sets of characters within the spans.
    """
    # If both annotators provided no annotations, consider it as a full agreement
    if not annotator1_spans and not annotator2_spans:
        return 1.0
    
    # Convert spans to sets of characters
    chars_in_annotator1 = set()
    for span in annotator1_spans:
        chars_in_annotator1.update(range(span[0], span[1]))
    
    chars_in_annotator2 = set()
    for span in annotator2_spans:
        chars_in_annotator2.update(range(span[0], span[1]))

    # Calculate intersection and union sizes
    intersection = chars_in_annotator1.intersection(chars_in_annotator2)
    union = chars_in_annotator1.union(chars_in_annotator2)

    # Calculate Dice Coefficient
    if len(union) == 0:  # Avoid division by zero
        return 0.0  # If only one annotator has no annotations, it's a disagreement
    else:
        return (2 * len(intersection)) / (len(chars_in_annotator1) + len(chars_in_annotator2))


def lcs_agreement_pairwise(annotations, text_length):
    """
    Calculate the inter-annotator agreement using the Longest Common Subsequence (LCS) method,
    returning the score for each pair of annotators.
    annotations is a list of lists, where each inner list contains the spans annotated by one annotator.
    text_length is the length of the text being annotated.
    """
    # Convert span annotations to binary sequence representations
    binary_annotations = []
    for annotator_spans in annotations:
        binary_sequence = [0] * text_length
        for start, end in annotator_spans:
            for i in range(start, end):
                binary_sequence[i] = 1
        binary_annotations.append(binary_sequence)

    # Calculate pairwise LCS agreement
    pairwise_agreements = {}
    for i in range(len(binary_annotations)):
        for j in range(i + 1, len(binary_annotations)):
            lcs_length = longest_common_subsequence(binary_annotations[i], binary_annotations[j])
            agreement = lcs_length / text_length
            pairwise_agreements[f"Annotator {i+1} & Annotator {j+1}"] = agreement

    return pairwise_agreements






