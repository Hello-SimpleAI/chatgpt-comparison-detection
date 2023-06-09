# -*- coding: utf-8 -*-

import argparse
import glob
import os
import time


_PARSER = argparse.ArgumentParser('dl eval')
_PARSER.add_argument(
    '-i', '--input', type=str, help='input file path',
    default='text/zh'
)
_PARSER.add_argument('-b', '--batch-size', type=int, default=64, help='batch size')
_PARSER.add_argument('--cuda', '-c', type=str, default='0', help='gpu ids, like: 1,2,3')
_PARSER.add_argument('--seed', '-s', type=int, default=42, help='random seed.')
_PARSER.add_argument('--max-length', type=int, default=512, help='max_length')
_PARSER.add_argument("--pair", action="store_true", default=False, help='paired input')
_PARSER.add_argument("--multisource", action="store_true", default=False, help='ood eval')

_ARGS = _PARSER.parse_args()

if len(_ARGS.cuda) > 1:
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

os.environ["OMP_NUM_THREADS"] = '8'
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # if cuda >= 10.2
os.environ['CUDA_VISIBLE_DEVICES'] = _ARGS.cuda

import torch
# import platform
# from pprint import pprint
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from transformers import pipeline
from datasets import Dataset, concatenate_datasets


def printf(*args):
    print(time.asctime(), "-", *args)


def predict_data(setting, lang, checkpoint, device, batch_size):
    if '-pair' in checkpoint:
        def func(batch):
            paired = [dict(text=q, text_pair=a) for q, a in zip(batch['question'], batch['answer'])]
            out = detector(paired , max_length=512, truncation=True)
            batch['pred'] = [int(o['label'][-1]) for o in out]
            return batch
    else:
        def func(batch):
            out = detector(batch['answer'], max_length=512, truncation=True)
            batch['pred'] = [int(o['label'][-1]) for o in out]
            # batch['prob'] = [o['score'] for o in out]
            return batch

    path = f"hc3/{setting}/{lang}_test.csv"  # path to the csv data from the google drive
    print('\n\n', path)
    test_df = pd.read_csv(path)
    dataset = Dataset.from_pandas(test_df)
    print(dataset)
    detector = pipeline('text-classification', model=checkpoint, device=device, framework='pt')
    dataset = dataset.map(func, batched=True, batch_size=batch_size, desc='test')
    return dataset


def evaluate_func(setting, lang, checkpoint, device, batch_size, sources):
    '''
    :param checkpoint: model saved dir
    '''
    def get_ds(_setting):
        path = os.path.join(checkpoint, _setting + '.test.json')
        if os.path.exists(path):
            ds = Dataset.from_json(path)
        else:
            ds = predict_data(_setting, lang, checkpoint, device, batch_size)
            ds.to_json(path, orient='records', lines=True, force_ascii=False)
            printf(path)
        return ds

    if 'mix' in setting:
        data = list()
        for mode in ('text', 'sent'):
            data.append(get_ds(setting.replace('mix', mode)))
        dataset = concatenate_datasets(data)
    else:
        dataset = get_ds(setting)

    detail = list()
    data_source = set(dataset['source'])
    assert set(sources) == data_source
    for s in sources:
        subset = dataset.filter(lambda x: x["source"] == s)
        report = classification_report(
            subset['label'], subset['pred'], target_names=['human','chatgpt'], output_dict=True
        )
        detail.extend([
            report['weighted avg']['f1-score'], report['chatgpt']['f1-score'], report['human']['f1-score']
        ])

    label = dataset['label']
    pred = dataset['pred']
    # prob = dataset['prob']
    clf_report = classification_report(
        label, pred, target_names=['human','chatgpt'], output_dict=True
    )
    auc = roc_auc_score(label, pred)
    # con_mat = confusion_matrix(label, pred)
    overall = [
        clf_report['accuracy'],
        auc,
        clf_report['weighted avg']['precision'],
        clf_report['weighted avg']['recall'],
        clf_report['weighted avg']['f1-score'],
        clf_report['chatgpt']['precision'],
        clf_report['chatgpt']['recall'],
        clf_report['chatgpt']['f1-score'],
        clf_report['human']['precision'],
        clf_report['human']['recall'],
        clf_report['human']['f1-score'],
        # clf_report['chatgpt']['support'],
        # clf_report['human']['support'],
    ]
    return overall, detail


def main(args: argparse.Namespace):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    name = args.input.replace('/', '-')
    lang = args.input[-2:]
    if lang == 'zh':
        sources = ['baike', 'finance', 'law', 'medicine', 'nlpcc_dbqa', 'open_questions', 'psychology']
    else:
        sources = ['finance', 'medicine', 'open_questions', 'reddit_eli5', 'wikipedia_csai']

    model_prefix = './models/' + args.input
    if args.pair:
        model_prefix += '-pair'

    if args.multisource:
        setting = args.input.split('/')[0]
        for source in sources:
            prefix = f"{model_prefix}-{source}"
            checkpoints = glob.glob(prefix + '/*')
            assert len(checkpoints) == 1
            printf(checkpoints[0])
            metrics = evaluate_func(setting, lang, checkpoints[0], device, args.batch_size, sources)
            for m, n in zip(metrics, ('multisource', 'multisource-source')):
                m = [i * 100 for i in m]
                row = [args.input, source] + m
                if args.pair:
                    n += '-pair'
                with open(f"ood/metrics/{n}.csv", mode='a') as file:
                    file.write(','.join(c if isinstance(c, str) else str(c) for c in row) + '\n')
                    printf('append to', file.name)
    else:
        checkpoints = glob.glob(model_prefix + '/*')
        checkpoints = [(c, int(c.split('/')[-1].split('-')[-1])) for c in checkpoints]
        ckpt = sorted(checkpoints, key=lambda x: x[1])[-1][0]  # max step saved
        printf(ckpt)
        for setting in ('mix', 'mix-filter'):  # ('text', 'text-filter', 'sent', 'sent-filter')
            metrics = evaluate_func(setting, lang, ckpt, device, args.batch_size, sources)
            for m, n in zip(metrics, (name, name + '-source')):
                m = [i * 100 for i in m]
                row = [args.input, setting] + m + [ckpt]
                if args.pair:
                    n += '-pair'
                with open(f"metrics/{n}.csv", mode='a') as file:
                    file.write(','.join(c if isinstance(c, str) else str(c) for c in row) + '\n')
                    printf('append to', file.name)


if __name__ == '__main__':
    with torch.no_grad():
        main(_ARGS)
