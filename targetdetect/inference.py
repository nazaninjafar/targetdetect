from models import GPTInference, OpenLLMInference
from prompt_dump import candidate_prompts
from data_loader import read_jsonl
import json
import os
from llm_util import parse_targets
import jsonlines
import random

import argparse
from tqdm import tqdm
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser(description='Target Detection Inference')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-1106', help='LLM model name')
    parser.add_argument('--prompt', type=int, default=0, help='Prompt index')
    parser.add_argument('--dataset', type=str, default='implicit-hate-corpus', help='Dataset name')
    parser.add_argument('--datamode', type=str, default='test', help='Dataset mode')
    return parser.parse_args()

def main():
    # read in data
    args = get_args()



    dataset = args.dataset 
    datamode = args.datamode
    random_samples = read_jsonl(f"data/{dataset}/{datamode}_hate.jsonl")

    llm_model_name = args.model

    
    if 'gpt-3.5' in llm_model_name:
        small_model_name = "gpt3.5"
        llm = GPTInference(llm_model_name)
    elif 'gpt-4' in llm_model_name:
        small_model_name = "gpt4"
        llm = GPTInference(llm_model_name)
    else:
        small_model_name = llm_model_name.split("/")[1]
        llm = OpenLLMInference(llm_model_name)
    selected_prompt = candidate_prompts[0]
    

    all_prompts = []
    outputs = []
    
    prompt_inst = selected_prompt
    for sample in tqdm(random_samples):
        text = sample["post"]
        prompt = prompt_inst + " Input: <" + text + ">"
        if "gpt" in llm_model_name:
            output = llm.get_model_predict(prompt)
            outputs.append(output)
        all_prompts.append(prompt)
    
    if "gpt" not in llm_model_name:
        outputs = llm.get_model_predict(all_prompts)
    
    # parse outputs to get targets 
    targets = parse_targets(outputs)

    # write targets to file
    outfilename = f"outputs/{dataset}_{small_model_name}_{datamode}.jsonl"
    with jsonlines.open(outfilename, "w") as writer:
        for i, sample in enumerate(random_samples):
            sample["targets"] = targets[i]
            writer.write(sample)



if __name__ == "__main__":
    main()
