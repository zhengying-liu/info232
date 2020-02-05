# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import sys
import os
import argparse

import pandas as pd

from importlib import import_module

from teacher import grade


def parse_args(main_description="Grading script to grade local answers only"):
    parser = argparse.ArgumentParser(description=main_description)
    parser.add_argument('TP', help='Which TP (directory) to grade', type=str)
    args = parser.parse_args()
    return args


def run(remote_name, tp='TP0'):
    answer = import_module('{}.answers'.format(tp))
    question = import_module('{}.questions'.format(tp))
    print("="*80)
    results, status = grade(question, answer)
    print("="*80)
    results['name'] = remote_name
    status['name'] = remote_name
    return results, status


def main():
    args = parse_args()
    TP = args.TP
    GRADE_DIR = 'GRADES'
    os.makedirs(GRADE_DIR, exist_ok=True)
    
    # FIXME ugly hack to access student code if they forgot to use relative import
    sys.path.append('./{}'.format(TP))

    scores, status = run('local', TP)
    all_scores = [scores]
    all_status = [status]
    score_table = pd.DataFrame(all_scores)
    status_table = pd.DataFrame(all_status)
    score_path = os.path.join(GRADE_DIR, '{}-scores_1.csv'.format(TP))
    status_path = os.path.join(GRADE_DIR, '{}-status_1.csv'.format(TP))
    score_table.to_csv(score_path)
    status_table.to_csv(status_path)



if __name__ == '__main__':
    main()
