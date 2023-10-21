#evaluating the classification performance with f1,precision and recall scores 
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score



def calculate_classification_metrics(preds,labels):
    f1 = f1_score(labels,preds,average='weighted')
    precision = precision_score(labels,preds,average='weighted')
    recall = recall_score(labels,preds,average='weighted')
    accuracy = accuracy_score(labels,preds)
    return f1,precision,recall,accuracy