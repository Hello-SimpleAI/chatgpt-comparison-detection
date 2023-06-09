"""
ml detectors.
"""

import argparse
import json
import os
import pickle
import random
import re
import time
from typing import Dict, List, Tuple

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from datasets import Dataset, concatenate_datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def printf(*args):
    print(time.asctime(), "-", *args)


# code borrowed from https://github.com/blmoistawinde/HarvestText
def sent_cut_zh(para: str) -> List[str]:
    para = re.sub('([。！？\?!])([^”’)\]）】])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{3,})([^”’)\]）】….])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…+)([^”’)\]）】….])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?!]|\.{3,}|\…+)([”’)\]）】])([^，。！？\?….])', r'\1\2\n\3', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    sentences = para.split("\n")
    sentences = [sent.strip() for sent in sentences]
    sentences = [sent for sent in sentences if len(sent.strip()) > 0]
    return sentences


def data_to_xy(dataset: Dataset) -> Tuple:
    return np.asarray(dataset['x']), np.asarray(dataset['label'])


def eval_func(model, setting, lang, test, sources):
    def read_ds(name):
        path = f"results/{name}-{lang}.test-{test}.test.json"
        return Dataset.from_json(path)

    # test run
    if 'mix' in setting:
        data = list()
        for mode in ('text', 'sent'):
            data.append(read_ds(setting.replace('mix', mode)))
        dataset = concatenate_datasets(data)
    else:
        dataset = read_ds(setting)

    x, y = data_to_xy(dataset)
    printf('predict test', setting)
    y_pred = model.predict(x)
    dataset = dataset.add_column('pred', y_pred)
    # y_prob = model.predict_proba(x)

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

    clf_report = classification_report(
        y, y_pred, target_names=['human','chatgpt'], output_dict=True
    )
    auc = roc_auc_score(y, y_pred)
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


def main(args: argparse.Namespace, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    name = f"ml-{args.input.replace('/', '-')}.test-{args.test}"
    lang = args.input[-2:]
    if lang == 'zh':
        sources = ['baike', 'finance', 'law', 'medicine', 'nlpcc_dbqa', 'open_questions', 'psychology']
    else:
        sources = ['finance', 'medicine', 'open_questions', 'reddit_eli5', 'wikipedia_csai']

    ckpt_path = f"results/{args.input.replace('/', '-')}.test-{args.test}.pkl"
    with open(ckpt_path, 'rb') as file:
        model = pickle.load(file)

    if args.test > 0:
        setting_names = ('text', 'text-filter', 'sent', 'sent-filter', 'mix', 'mix-filter')
    else:
        setting_names = ('text', 'text-filter')

    for setting in setting_names:
        metrics = eval_func(model, setting, lang, args.test, sources)
        for m, n in zip(metrics, ('metric_ml', 'metric_ml' + '-source')):
            m = [i * 100 for i in m]
            row = [args.input, setting] + m + [ckpt_path]
            with open(f"metrics/{n}.csv", mode='a') as file:
                file.write(','.join(c if isinstance(c, str) else str(c) for c in row) + '\n')
                printf('append to', file.name)

    return


if __name__ == '__main__':
    _PARSER = argparse.ArgumentParser('detector')
    _PARSER.add_argument(
        '-i', '--input', type=str, help='input file path',
        default='text/zh'
    )
    _PARSER.add_argument(
        '-t', '--test', type=int, default=1, help='test no. (0: ppl, 1: rank bucket)'
    )

    _ARGS = _PARSER.parse_args()
    if os.path.basename(_ARGS.input)[-2:] == 'en':
        from nltk.data import load as nltk_load

        # https://huggingface.co/Hello-SimpleAI/chatgpt-detector-ling/resolve/main/english.pickle
        NLTK = nltk_load("data/english.pickle")
        sent_cut = NLTK.tokenize
    else:
        sent_cut = sent_cut_zh

    main(_ARGS)
