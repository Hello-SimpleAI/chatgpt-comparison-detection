"""
Script for training models on a single domain for the OOD setting.

pip install datasets evaluate scikit-learn torch==1.12.1 transformers
"""

import argparse
import os
import random

_PARSER = argparse.ArgumentParser('dl detector')
_PARSER.add_argument(
    '-i', '--input', type=str, help='input file path',
    default='text/zh'
)
_PARSER.add_argument(
    '-s', '--source', type=str, help='model name', default='all'
)
_PARSER.add_argument(
    '-m', '--model-name', type=str, help='model name', default='hfl/chinese-roberta-wwm-ext'
)
_PARSER.add_argument('-b', '--batch-size', type=int, default=16, help='batch size')
_PARSER.add_argument('-e', '--epochs', type=int, default=2, help='batch size')
_PARSER.add_argument('--cuda', '-c', type=str, default='0', help='gpu ids, like: 1,2,3')
_PARSER.add_argument('--seed', type=int, default=42, help='random seed.')
_PARSER.add_argument('--max-length', type=int, default=512, help='max_length')
_PARSER.add_argument("--pair", action="store_true", default=False, help='paired input')


_ARGS = _PARSER.parse_args()

if len(_ARGS.cuda) > 1:
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

os.environ["OMP_NUM_THREADS"] = '8'
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # if cuda >= 10.2
os.environ['CUDA_VISIBLE_DEVICES'] = _ARGS.cuda


def main(args: argparse.Namespace):
    import numpy as np
    from datasets import Dataset, concatenate_datasets
    import evaluate
    import pandas as pd
    import torch
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
        Trainer, TrainingArguments
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    def read_train_test(name):
        print(name)
        prefix = 'hc3/'  # path to the csv data from the google drive
        train_df = pd.read_csv(os.path.join(prefix, name + '_train.csv'))
        test_df = pd.read_csv(os.path.join(prefix, name + '_test.csv'))
        len(train_df)
        len(test_df)
        print('train', train_df['source'].value_counts())
        print('test', test_df['source'].value_counts())
        if args.source != 'all':
            train_df = train_df[train_df.source == args.source]
            test_df = test_df[test_df.source == args.source]
        print(train_df.head())
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)
        print(train_dataset)
        print(test_dataset)
        return train_dataset, test_dataset

    if 'mix' in args.input:
        data = [read_train_test(args.input.replace('mix', m)) for m in ('text', 'sent')]
        train_dataset = concatenate_datasets([data[0][0], data[1][0]])
        test_dataset = concatenate_datasets([data[0][1], data[1][1]])
    else:
        train_dataset, test_dataset = read_train_test(args.input)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    kwargs = dict(max_length=args.max_length, truncation=True)
    if args.pair:
        def tokenize_fn(example):
            return tokenizer(example['question'], example['answer'], **kwargs)
    else:
        def tokenize_fn(example):
            return tokenizer(example['answer'], **kwargs)

    print('Tokenizing and mapping...')
    train_dataset = train_dataset.map(tokenize_fn)
    if test_dataset is not None:
        test_dataset = test_dataset.map(tokenize_fn)

    # remove unused columns
    names = ['id', 'question', 'answer', 'source']
    tokenized_train_dataset = train_dataset.remove_columns(names)
    if test_dataset is not None:
        tokenized_test_dataset = test_dataset.remove_columns(names)
    else:
        tokenized_test_dataset = None
    print(tokenized_train_dataset)

    accuracy = evaluate.load("accuracy")
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=predictions, references=labels)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    output_dir = "./models/"  + args.input  # checkpoint save path
    if args.pair:
        output_dir += '-pair'
    if args.source != 'all':
        output_dir += '-' + args.source

    training_args = TrainingArguments(
        output_dir=output_dir,
        seed=args.seed,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy='no' if test_dataset is None else 'epoch',
        save_strategy='epoch',
        save_total_limit=1
        # load_best_model_at_end=True,
        # metric_for_best_model='accuracy'
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()


if __name__ == '__main__':
    if _ARGS.input.endswith('zh'):
        _ARGS.model_name = 'hfl/chinese-roberta-wwm-ext'
        _ARGS.epochs = 2
    else:
        _ARGS.model_name = 'roberta-base'
        _ARGS.epochs = 1

    main(_ARGS)
