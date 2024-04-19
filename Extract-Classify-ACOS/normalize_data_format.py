import re
from ftfy import fix_text
from underthesea import text_normalize, word_tokenize
import pandas as pd
from transformers import AutoTokenizer
from argparse import ArgumentParser

sentiment2id = {
    'positive': 0,
    'negative': 1,
    'neutral': 2
}

def process(text, tokenizer, do_segment=False):
  if do_segment:
    text = text_normalize(text)
    text = word_tokenize(text, format='text')
  return ' '.join(tokenizer.convert_ids_to_tokens(tokenizer(text)['input_ids'])[1:-1])

def process_label(text, quad, tokenizer, do_segment):
    quad = re.sub(r'[{}]', '', quad).strip().split(',')
    category, aspect, sentiment, opinion = quad
    category = category.replace('&', '_')
    # Split text into words
    words = text.split()
    aspect = process(aspect, tokenizer, do_segment)
    opinion = process(opinion, tokenizer, do_segment)
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

def normalize_format(path, name, subset, tokenizer, do_segment=False):
    with open(path) as f:
        data = f.read().split('\n\n')
        ids = []
        texts = []
        all_labels = []
        for example in data:
            example = example.split('\n')
            id = example[0]
            text = example[1]


            labels = example[2].split('; ')
            
            text = process(text, tokenizer, do_segment)

            ids.append(id)
            texts.append(text)
            all_labels.append([process_label(text, label, tokenizer, do_segment) for label in labels])

    
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

                f.write(f'{text}####{aspect_span} {opinion_span}\t{category}#{sentiment}\n')
              

def main(args=None):
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_texts, all_train_labels = normalize_format(f'{args.data_dir}/Train.txt', args.experiment_name, 'train', tokenizer, args.do_segment)
    dev_texts, all_dev_labels = normalize_format(f'{args.data_dir}/Dev.txt', args.experiment_name, 'dev', tokenizer, args.do_segment)
    test_texts, all_test_labels = normalize_format(f'{args.data_dir}/Test.txt', args.experiment_name, 'test', tokenizer, args.do_segment)

    create_test_pair(train_texts, all_train_labels, args.experiment_name, 'train')
    create_test_pair(dev_texts, all_dev_labels, args.experiment_name, 'dev')
    create_test_pair(test_texts, all_test_labels, args.experiment_name, 'test')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name',
                        type=str,
                        required=True)
    parser.add_argument('--do_segment',
                        action='store_true',
                        required=False)
    parser.add_argument('--data_dir',
                        type=str,
                        required=True)
    parser.add_argument('--output_dir',
                        type=str,
                        required=False,
                        default='tokenized_data')
    parser.add_argument('--experiment_name',
                        type=str,
                        default='uit_absa_res')

    args = parser.parse_args()
    
    main(args)