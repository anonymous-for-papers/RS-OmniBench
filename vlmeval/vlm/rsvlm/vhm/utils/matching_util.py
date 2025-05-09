import string
import copy as cp
import os
from vlmeval.vlm.rsvlm.vhm.smp import *


def can_infer_option(answer, num_choice=5):
    verbose = os.environ.get('VERBOSE', 0)
    choices = string.ascii_uppercase[:num_choice]
    if 'Failed to obtain answer via API' in answer:
        return False

    bard_err = [
        "Sorry, I can't help with images of people yet.",
        "I can't process this file."
    ]
    for err in bard_err:
        if err in answer:
            return 'E'

    def count(splits, choices='ABCDE', prefix='', suffix=''):
        cnt = 0
        for c in choices:
            if prefix + c + suffix in splits:
                cnt += 1
        return cnt

    answer_mod = cp.copy(answer)
    chars = '.()[],:;!*#{}'
    for c in chars:
        answer_mod = answer_mod.replace(c, ' ')

    splits = [x.strip() for x in answer_mod.split()]

    if count(splits, choices) == 1:
        for ch in choices:
            if 'A' in splits and len(splits) > 3 and verbose:
                double_log(f'A might be a quantifier in the string: {answer}. ', fout)
                return False
            if ch in splits:
                return ch
    return False


def can_infer_text(answer, choices):
    answer = answer.lower()
    assert isinstance(choices, dict)
    for k in choices:
        assert k in 'ABCDE'
        choices[k] = str(choices[k]).lower()
    cands = []
    for k in choices:
        if choices[k] in answer:
            cands.append(k)
    if len(cands) == 1:
        return cands[0]
    return False


def can_infer(answer, choices):
    answer = str(answer)
    copt = can_infer_option(answer)
    return copt if copt else can_infer_text(answer, choices)
