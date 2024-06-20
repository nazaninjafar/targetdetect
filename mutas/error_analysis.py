import jsonlines
from transformers import RobertaTokenizerFast
from data_loader import convert_to_bio, tokenize_and_align_labels_2
from data_util import convert_to_span,convert_predicted_to_spans,convert_tensor_to_spans


path = 'missclassified_samples_implicit-hate-corpus_trained_mutas_implicit-hate-corpus_roberta-large_implicit-hate-corpus.jsonl'

# Load the missclassified samples
with jsonlines.open(path) as reader:
    missclassified_samples = [sample for sample in reader]

#remove empty dict samples 

missclassified_samples = [sample for sample in missclassified_samples if sample.get("context")]



tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large',add_prefix_space=True)
# Convert the missclassified samples to bio tags



def get_boundary_errors(sample):
    #check if the missed tokens are inside any of the truth tokens
    boundary_error_num = 0
    total_truth_tokens = 0
    true_spans  = sample['true_spans']
    predicted_spans = sample['predicted_spans']
    missed_spans = sample['missed_spans']
    for missed_span in predicted_spans:
        for true_span in true_spans:
            
            if missed_span[0] == true_span[0] and missed_span[-1] == true_span[-1]:
                continue
            else:
                if missed_span[0] >= true_span[0] and missed_span[-1] <= true_span[-1]:
                    boundary_error_num += 1
    total_truth_tokens += len(true_spans)
               
            
    return boundary_error_num,total_truth_tokens


all_boundary_errors = 0
all_total_truth_tokens = 0
boundry_probs = []
num_predicted =0
for sample in missclassified_samples:
    if len(sample['predicted_tokens'])==len(sample['true_tokens']):
        num_predicted += 1

    boundary_error_num,total_truth_tokens = get_boundary_errors(sample)
    all_boundary_errors += boundary_error_num
    all_total_truth_tokens += total_truth_tokens
    if boundary_error_num > 0:
        boundry_probs.append(1)

print("all_boundary_errors",all_boundary_errors)
print("all_total_truth_tokens",all_total_truth_tokens)
print("boundry_probs",len(boundry_probs)/len(missclassified_samples))

print("num_predicted",num_predicted/len(missclassified_samples))



    




