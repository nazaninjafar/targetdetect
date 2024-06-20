from vllm import LLM, SamplingParams
import os
from tqdm import tqdm
import numpy as np 
import pandas as pd
from fuzzywuzzy import fuzz
os.environ['TRANSFORMERS_CACHE'] = ''



def parse_targets(outputs):
    all_targets = []
    for output in outputs:
        print(output)
        if 'Output:' in output:
            targets = output.split('Output:')[1].strip()
            targets = targets.split('\n')[0].strip()
            targets = targets.replace('[','').replace(']','').replace('"','').replace("'",'').split(',')
            #separate any target that has \n in it
            for target in targets:
                if '\n' in target:
                    target = target.split('\n')[0].strip()

            targets = [target.strip() for target in targets]
            all_targets.append(targets)
        else:
            #check if there is any string that looks like python list
            if '[' in output and ']' in output:
                targets = output.split('[')[1].split(']')[0].replace('"','').replace("'",'').split(',')
                #separate any target that has \n in it
                for target in targets:
                    if '\n' in target:
                        target = target.split('\n')[0].strip()
                targets = [target.strip() for target in targets]
                all_targets.append(targets)
            else:
                # check if there is a newline character
                if '\n' in output:
                    targets = output.split('\n')[0].strip()
                    all_targets.append([targets])
                else:
                    all_targets.append([])
    return all_targets

            


def get_llm_engine(model_id = "meta-llama/Llama-2-13b-hf",download_dir = ''):
    
    sampling_params = SamplingParams(temperature=0.0,max_tokens= 200)
    #if model is mistral or falcon the gpu does not have compute capability so change the dtype to 'float16'

    llm = LLM(model=model_id,download_dir = download_dir,dtype= 'float16',trust_remote_code=True)

    return llm, sampling_params



def target_classification_prompter(dataset,targets,pretrained_model= None,personalized = False):
    prompts = []
    if personalized:
        pass 

    init_instruction=f"""Your task is to identify which protected group these posts are targeting. The Input_Post is given in <> and you should output the protected group.
    Choose the protected group from the following list: [{targets}]. If you think the post is not targeting any of these groups, output 'none'.
    Example: Input_Post: <i wonder if every white person investigated today including potus is because he is white . in today's world being white is crime unto itself>
    Target: white
    """
    for data in tqdm(dataset):
        text = data['post']
        
        prompt = f"""{init_instruction} Input_Post: <{text}>"""
        
        prompts.append(prompt)
    
    return prompts

def get_llm_outputs_target_classification(prompts):
    llm, sampling_params = get_llm_engine() 
    outputs = llm.generate(prompts, sampling_params)
    return outputs


def parse_llm_outputs_target_classification(outputs):
    preds = []
    for output in outputs:
        out=output.outputs[0].text
        
        if 'Target:' in out:
            
            pred = out.split('Target:')[1].strip()
            if len(pred)<=1:  
                pred = 'none'
                
        else:
            pred = 'none'
        #clean the pred and remove the punctuation, strip from spaces, and if there is Input_post, split it and take before this
        pred = pred.replace('.','').replace(',','').replace('?','').replace('!','').replace(';','').strip()
        #remove \n and \t
        if '\n' in pred:
            pred = pred.split('\n')[0]
        if '\t' in pred:
            pred = pred.split('\t')[0]

        
        if 'Input_Post:' in pred:
            pred = pred.split('Input_Post:')[0].strip()
            
        #remove people from pred
        if 'people' in pred:
            pred = pred.replace('people','').strip()

        preds.append(pred)

        
    return preds




def are_words_similar(a,b,threshold = 60):
    if a == b:
        return True
    if a in b or b in a:
        return True
    if fuzz.ratio(a,b) >= threshold:
        return True
    
    
    
#     return False

    
