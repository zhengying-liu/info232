# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

from .evaluation import ensure_score
from .evaluation import evaluate

def Score(sc):
    def score_decorator(question):
        question.score = sc
        question.max_score = sc
        return question
    return score_decorator


def SubQuestion(name):
    def question_decorator(question):
        question.__question_name__ = name
        return question
    return question_decorator

