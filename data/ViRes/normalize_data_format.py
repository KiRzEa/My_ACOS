import re
from ftfy import fix_text
from underthesea import normalize_text
import pandas as pd



sentiment2id = {
    'positive': 0,
    'negative': 1,
    'neutral': 2
}

def process_label(text, quad, tokenizer):
    quad = re.sub(r'[{}]', '', quad).strip().split(',')
    category, aspect, sentiment, opinion = quad
    
    # Split text into words
    words = text.split()
    aspect = ' '.join(tokenizer(aspect)[0].tokens[1:-1])
    opinion = ' '.join(tokenizer(opinion)[0].tokens[1:-1])

    # Find aspect in text
    aspect_start_index = None
    aspect_end_index = None
    is_implicit_aspect = True
    for i in range(len(words)):
        if ' '.join(words[i:i+len(aspect.split())]) == aspect:
            aspect_start_index = i
            aspect_end_index = i + len(aspect.split())
            is_implicit_aspect = False
            break

    if is_implicit_aspect:
        # Find aspect in text
        aspect_start_index = -1
        aspect_end_index = -1

    # Find opinion in text
    opinion_start_index = None
    opinion_end_index = None
    is_implicit_opinion = True
    for i in range(len(words)):
        if ' '.join(words[i:i+len(opinion.split())]) == opinion:
            opinion_start_index = i
            opinion_end_index = i + len(opinion.split())
            is_implicit_opinion = False
            break

    if is_implicit_opinion:
        # Find opinion in text
        opinion_start_index = -1
        opinion_end_index = -1

    # Construct aspect and opinion spans
    aspect_span = f"{aspect_start_index},{aspect_end_index}"
    opinion_span = f"{opinion_start_index},{opinion_end_index}"

    return f"{aspect_span} {category} {sentiment2id[sentiment]} {opinion_span}"

def normalize_format(path, name, subset):
    with open(path) as f:
        data = f.read().split('\n\n')
        ids = []
        texts = []
        all_labels = []
        for example in data:
            example = example.split('\n')
            id = example[0]
            text = ' '.join(tokenizer(example[1])[0].tokens[1:-1])
            labels = example[2].split('; ')

            ids.append(id)
            texts.append(text)
            all_labels.append([process_label(text, label, tokenizer) for label in labels])

    
    with open(f'../Extract-Classify-ACOS/tokenized_data/{name}_{subset}_quad_bert.tsv', 'w') as f:
        for text, labels in zip(texts, all_labels):
            labels = '\t'.join(labels)
            f.write(f'{text}\t{labels}')


def main():
    normalize_format('Train.txt', 'uit_absa_res', 'train')
    normalize_format('Dev.txt', 'uit_absa_res', 'dev')
    normalize_format('Test.txt', 'uit_absa_res', 'test')

if __name__ == '__main__':
    main()