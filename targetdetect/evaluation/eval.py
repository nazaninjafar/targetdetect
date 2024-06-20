#evaluating the classification performance with f1,precision and recall scores 
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score

from evaluate import load
exact_match_metric = load("exact_match")

#calcuate exact match score 
def calculate_exact_match_score(preds,labels):
    max_length = max(len(preds),len(labels))
    correct = 0
    for i in range(max_length):
        if i >= len(preds):
            break
        if i >= len(labels):
            break
        results = exact_match_metric.compute(predictions=[preds[i]], references=[labels[i]])
        correct += results["exact_match"]
    return correct/len(preds)


def calculate_partial_match_score(preds,labels):
    max_length = max(len(preds),len(labels))
    correct = 0
    for i in range(max_length):
        if i >= len(preds):
            break
        if i >= len(labels):
            break
        tokenize_preds = preds[i].split(' ')
        tokenize_labels = labels[i].split(' ')
        current_correct = 0
        for word in tokenize_preds:
            if word in tokenize_labels:
                current_correct += 1
        correct += current_correct/len(tokenize_preds)
    return correct/len(preds)


#normalize text in a list 
def clean_text(text):
    text = text.lower()
    text = text.replace('\n','')
    text = text.replace('\t','')
    #remove space at the beginning and end of the text
    text = text.strip()

    return text


#calculate f1 precision and recall scores based on retrieved and relevant spans
def calculate_precision(retrieved,relevant):

    if len(retrieved) == 0:
        return 0
    else:
        TPs = 0
        for span in retrieved:
            if span in relevant:
                TPs += 1
        
        return  TPs/len(retrieved)
    
def calculate_recall(retrieved,relevant):
    if len(relevant) == 0:
        return 0
    else:
        TPs = 0
        for span in retrieved:
            if span in relevant:
                TPs += 1
        return TPs/len(relevant)
    

def calculate_f1(precision,recall):
    if precision == 0 and recall == 0:
        return 0
    else:
        return 2*precision*recall/(precision+recall)
    

