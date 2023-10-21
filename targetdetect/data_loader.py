import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import jsonlines 
import nltk
import torch
from nltk.stem import WordNetLemmatizer
import inflect
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
nltk.download('wordnet')

# Create a WordNet lemmatizer
lemmatizer = WordNetLemmatizer()


p = inflect.engine()

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







class ImplicitHateCorpus:
    def __init__(self, data_path = 'data/implicit-hate-corpus/'):
        self.data_path = data_path
        
        
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



        



        


        
        

