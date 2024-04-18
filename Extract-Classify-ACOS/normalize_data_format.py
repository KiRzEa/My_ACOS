import re
from ftfy import fix_text
from underthesea import text_normalize
import pandas as pd
from transformers import AutoTokenizer


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

def normalize_format(path, name, subset, tokenizer):
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

    
    with open(f'tokenized_data/{name}_{subset}_quad_bert.tsv', 'w') as f:
        for text, labels in zip(texts, all_labels):
            labels = '\t'.join(labels)
            f.write(f'{text}\t{labels}\n')
    
    return texts, all_labels

def create_test_pair(texts, all_labels, name, subset):
    with open(f'tokenized_data/{name}_{subset}_pair.tsv', 'w') as f:
        for text, labels in zip(texts, all_labels):
            for label in labels:
                components = label.split()
                aspect_span, category, sentiment, opinion_span = tuple(components)

                f.write(f'{text}####{aspect_span} {opinion_span} {category}#{sentiment}\n')
              

def main():
    
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-multilingual-uncased')

    name = 'uit_absa_res'

    train_texts, all_train_labels = normalize_format('../data/ViRes/Train.txt', name, 'train', tokenizer)
    dev_texts, all_dev_labels = normalize_format('../data/ViRes/Dev.txt', name, 'dev', tokenizer)
    test_texts, all_test_labels = normalize_format('../data/ViRes/Test.txt', name, 'test', tokenizer)

    create_test_pair(train_texts, all_train_labels, name, 'train')
    create_test_pair(dev_texts, all_dev_labels, name, 'dev')
    create_test_pair(test_texts, all_test_labels, name, 'test')

if __name__ == '__main__':
    main()