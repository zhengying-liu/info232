# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import inspect

QUESTION_PREFIX = "question"
ANSWER_PREFIX   = "answer"

def is_answer(name, obj):
    return name.startswith(ANSWER_PREFIX)

def is_question(name, obj):
    try:
        return obj.__question_name__.startswith(QUESTION_PREFIX)
    except AttributeError:
        return name.startswith(QUESTION_PREFIX)


def extract_answers(module):
    module_members = inspect.getmembers(module)
    answers = dict(filter(lambda e : is_answer(*e), module_members) )
    return answers

def extract_questions(module):
    module_members = inspect.getmembers(module)
    questions = dict(filter(lambda e : is_question(*e), module_members) )
    return questions

def answer_name(question_name):
    n = len(QUESTION_PREFIX)
    suffix = question_name[n:]
    name = ANSWER_PREFIX + suffix
    return name

def find_answer(question, all_answers):
    question_name = question.__name__
    answer = all_answers.get(answer_name(question_name), None)
    if answer is not None :
        return answer
    else:
        try :
            question_name = question.__question_name__
            answer = all_answers.get(answer_name(question_name), None)
        except AttributeError:
            pass
        return answer
