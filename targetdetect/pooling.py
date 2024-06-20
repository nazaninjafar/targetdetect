import jsonlines 

import pandas as pd



def preprocess_text(text):
    text = text.lower()
    text = text.replace("@","")
    text = text.replace("#","")
    text = text.replace("<","")
    text = text.replace(">","")
    text = text.replace("'","")

    return text

def score_targets(gt_targets,pred_targets):
    tp = 0
    fp = 0
    fn = 0
    for target in gt_targets:
        target_splits = target.split(" ")
        pred_target_splits = [t.split(" ") for t in pred_targets]
        
        if target in pred_targets:
            tp += 1
        #el if any target splits are in pred_targets
        
        else:
            fn += 1
    for target in pred_targets:
        if target not in gt_targets:
            fp += 1
    return tp,fp,fn


def sanity_check(text,targets):
    
    for target in targets:
        if target not in text:
        
            return False
    return True

def get_substring(text,targets):
    ranges = []
    for target in targets:
        start = text.find(target)
        end = start + len(target)
        ranges.append((start,end))
    return ranges

def pred_recall(gt_ranges, pred_ranges, gts, preds,text):
    tp = 0
    if len(gt_ranges) == 0 and len(pred_ranges) == 0:
        return 1,0
    elif len(gt_ranges) == 0 and len(pred_ranges) != 0:
        return 0,1
    for i, gt_range in enumerate(gt_ranges):
        
        gt = gts[i]
        pred_range = None
      
        for pr in pred_ranges:
            if pr[0] == gt_range[0] and pr[1] == gt_range[1]:
                pred_range = pr
                break
        if pred_range is not None:
            tp += 1
        else:
            
            for pr in pred_ranges:
                if pr[0] <= gt_range[0] and pr[1] <= gt_range[1]:
                    pred_range = pr
                    break
            if pred_range is not None:


                pred_words = text[pred_range[0]:pred_range[1]].split()
               
                gt_words = gts[i].split()
                matching_words = set(pred_words) & set(gt_words)
                tp += (len(matching_words) / len(gt_words))
                
          

    tp = tp / len(gt_ranges)
    
    return tp

def pred_precision(gt_ranges, pred_ranges, gts, preds,text):
    tp = 0
   
    if len(gt_ranges) == 0 and len(pred_ranges) == 0:
        return 1
    elif len(gt_ranges) == 0 and len(pred_ranges) != 0:
        return 0
    elif len(gt_ranges) != 0 and len(pred_ranges) == 0:
        return 0
    for i, pred_range in enumerate(pred_ranges):
        pred = preds[i]
        gt_range = None
        for gr in gt_ranges:
            if gr[0] == pred_range[0] and gr[1] == pred_range[1]:
                gt_range = gr
                break
        if gt_range is not None:
            tp += 1
        else:
           
            for gr in gt_ranges:
                if gr[0] <= pred_range[0] and gr[1] <= pred_range[1]:
                    gt_range = gr
                    break
            if gt_range is not None:
                gt_words = text[gt_range[0]:gt_range[1]].split()
                pred_words = preds[i].split()
                matching_words = set(pred_words) & set(gt_words)
                tp += (len(matching_words) / len(pred_words))
                
            
            

    tp = tp / len(pred_ranges)
    return tp



def eval_llms(filename = filename):
    with jsonlines.open(filename) as reader:
        outputs = list(reader)

    F1_score =0
    precision_score = 0
    recall_score = 0
    for output in outputs:
        text = output["text"]
        pred_targets = output["targets"]
        gt_targets = df[df["post"] == text]["target"].values.tolist()[0]
        lower = lambda x: x.lower()
        gt_targets = list(map(lower,gt_targets))
        pred_targets = list(map(lower,pred_targets))
        
        processed_text = preprocess_text(text)
 
        gold_substr_ranges = get_substring(processed_text,gt_targets)
        pred_substr_ranges = get_substring(processed_text,pred_targets)
       

        pred_targets = [target.lower() for target in pred_targets]
        
        
        if not sanity_check(processed_text,pred_targets):
            
            continue
        t_rec = pred_recall(gold_substr_ranges,pred_substr_ranges,gt_targets,pred_targets,processed_text)
        t_prec = pred_precision(gold_substr_ranges,pred_substr_ranges,gt_targets,pred_targets,processed_text)

        recall_score += t_rec
        precision_score += t_prec
        print("recall score: ",recall_score)
        print("precision score: ",precision_score)

    recall_score = recall_score / len(outputs)
    precision_score = precision_score / len(outputs)
    F1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score)
    print("F1 score: ",F1_score)

        

       
       
    
filename = ""    
eval_llms(filename = filename)
