# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

from .grab import extract_answers
from .grab import extract_questions
from .grab import find_answer


def grade(question_module, answer_module):
    all_questions = extract_questions(question_module)
    all_answers = extract_answers(answer_module)
    results, status = evaluate_all(all_questions, all_answers)
    print("SCORE = {} / {} ".format(results['total'], results['max_score']))
    return results, status


def evaluate_all(all_questions, all_answers):
    total_score = 0
    max_score = sum([ensure_score(q).max_score for q in all_questions.values()])
    results = dict(max_score=max_score)
    all_status = dict()
    for question_name, question in all_questions.items():
        answer = find_answer(question, all_answers)
        question = evaluate(question, answer)
        results[question_name] = question.score
        all_status[question_name] = question.status
        total_score += question.score
    results['total'] = total_score
    return results, all_status


NOTFOUND = "NOTFOUND"
PARTIAL  = "PARTIAL"
SUCCESS  = "SUCCESS"
ERROR    = "ERROR"
FAIL     = "FAIL"

def _print_eval_result(question, exception=""):
    print("{:9} [{}] ({}/{}) {}".format(question.status
        , question.__name__
        , question.score
        , question.max_score
        , exception))

def evaluate(question, answer):
    ensure_score(question)
    question.status = ""
    zero_score = 0
    if answer is None:
        question.status = NOTFOUND
        question.score  = zero_score
        _print_eval_result(question)
    else:
        try:
            question(answer)
            # FIXME : this assert will be catch like if question(answer) raised assert ...
            #  but it is different and should be handle differently
            assert question.score <= question.max_score, \
                'score for {} > max_score. score={} and max_score={}'.format(question.__name__, question.score, question.max_score)
            # if question(answer) evaluated sub question we handle it here
            if question.score <= 0:
                question.status = FAIL
                _print_eval_result(question)
            elif question.score < question.max_score :
                question.status = PARTIAL
                _print_eval_result(question)
            else:
                question.status = SUCCESS
                _print_eval_result(question)
        except AssertionError as ex:
            question.status = FAIL
            question.score  = zero_score
            _print_eval_result(question, ex)
        except Exception as ex:
            question.status = ERROR
            question.score  = zero_score
            _print_eval_result(question, ex)
    return question


def ensure_score(question):
    try:
        question.score
    except AttributeError:
        question.score =  1
    try:
        question.max_score
    except AttributeError:
        question.max_score = question.score
    return question
