import jsonlines
import re

def find_all_occurrences_regex(text, word):
    escaped_word = re.escape(word)
    return [match.start() for match in re.finditer(escaped_word, text)]


def convert_spaced_words(sentence):
    # Find all occurrences of spaced letters
    spaced_words = re.findall(r'(\b(?:[a-zA-Z] ){2,}[a-zA-Z]\b)', sentence)

    for word in spaced_words:
        # Remove spaces from each spaced word
        corrected_word = word.replace(" ", "")
        # Replace the spaced word in the original sentence with the corrected word
        sentence = sentence.replace(word, corrected_word)

    return sentence

def remove_extra_spaces(text):
    # Replace multiple spaces with a single space
    return re.sub(r'\s+', ' ', text).strip()

def convert_to_span(context,targets):
    context = context.lower()

    targets = [target.lower() for target in targets]
    # targets = [convert_spaced_words(target) for target in targets]
    #remove empty targets
    targets = [target for target in targets if target != '']
    target_spans = []
    failed_targets = []
    unique_targets = list(set(targets))
    for target in unique_targets: 

        target_occurrences = find_all_occurrences_regex(context,target)
 
        for start in target_occurrences:
            if start == -1:
                failed_targets.append(target)
                continue
            end = start + len(target)
            target_spans.append((start,end))
            
    return target_spans, failed_targets,context



def read_jsonl_file(path):
    data = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            data.append(obj)
    return data


def convert_predicted_to_spans(predicted):
    spans = []
    span = []
    for i,token_class in enumerate(predicted):
        if token_class == 'B-SPAN':
            span.append(i)
        elif token_class == 'I-SPAN':
            span.append(i)
        else:
            if len(span) > 0:
                spans.append(span)
                span = []
    if len(span) > 0:
        spans.append(span)
        span = []
    return spans
def convert_tensor_to_spans(tensor):
    spans = []
    span = []
    for i,token_class in enumerate(tensor):
        token_class = token_class.item()
        #each span starts with 1 and ends with 2(followed by 0)
        if token_class == 1:
            span.append(i)
        elif token_class == 2:
            span.append(i)
        else:
            if len(span) > 0:
                spans.append(span)
                span = []
    if len(span) > 0:
        spans.append(span)
        span = []
    return spans