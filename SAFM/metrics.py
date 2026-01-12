import collections
import os
import logging
import json
import string
import re
import numpy as np
from contextlib import closing
from multiprocessing import Pool, cpu_count
from subprocess import Popen, PIPE

from sacrebleu import corpus_bleu

AGG_OPS = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
COND_OPS = ['=', '>', '<', 'OP']
COND_OPS = ['=', '>', '<']

from dictdiffer import diff
from collections import defaultdict





def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)







def split_sentences(txt, splitchar=".", include_splitchar=False):
    """Split sentences of a text based on a given EOS char."""
    out = [s.split() for s in txt.strip().split(splitchar) if len(s) > 0]
    return out



def score(answer, gold):
    if len(gold) > 0:
        gold = set.union(*[simplify(g) for g in gold])
    answer = simplify(answer)
    tp, tn, sys_pos, real_pos = 0, 0, 0, 0
    if answer == gold:
        if not ('unanswerable' in gold and len(gold) == 1):
            tp += 1
        else:
            tn += 1
    if not ('unanswerable' in answer and len(answer) == 1):
        sys_pos += 1
    if not ('unanswerable' in gold and len(gold) == 1):
        real_pos += 1
    return np.array([tp, tn, sys_pos, real_pos])


def simplify(answer):
    return set(''.join(c for c in t if c not in string.punctuation) for t in answer.strip().lower().split()) - {'the', 'a', 'an', 'and', ''}


def computeBLEU(outputs, targets):
    targets = [[t[i] for t in targets] for i in range(len(targets[0]))]
    return corpus_bleu(outputs, targets, lowercase=True).score      # attention, lowercase=True


def compute_metrics(data, rouge=False, bleu=False, 

                            ):
    greedy = [datum[0] for datum in data]  
    answer = [datum[1] for datum in data]

    metric_keys = []
    metric_values = []

    if rouge:
        rouge = computeROUGE(greedy, answer)
        metric_keys += ['rouge1', 'rouge2', 'rougeL', 'avg_rouge']
        avg_rouge = (rouge['rouge_1_f_score'] + rouge['rouge_2_f_score'] + rouge['rouge_l_f_score']) / 3
        metric_values += [rouge['rouge_1_f_score'], rouge['rouge_2_f_score'], rouge['rouge_l_f_score'], avg_rouge]
    if bleu:
        bleu = computeBLEU(greedy, answer)
        metric_keys.append('bleu')
        # metric_values.append(bleu)
        metric_values.append(round(bleu, 2))
    metric_dict = collections.OrderedDict(list(zip(metric_keys, metric_values)))
    return metric_dict
