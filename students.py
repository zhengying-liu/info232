# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import subprocess

import pandas as pd

from git_call import git_remote_add_info232

STUDENTS = {
'victor-estrade': 'Estrade Victor',
# 'Didayolo':  'Pavao Adrien',
'herilalaina': 'Rakotoarison Herilalaina',
# 'fake_student':  'No Git Found test',
}


def clear_remote():
    truc = subprocess.check_output(['git', 'remote'])
    all_remotes = [e.decode("utf-8") for e in truc.split()]
    all_remotes.remove('origin')
    all_remotes.remove('upstream')
    for remote in all_remotes:
        subprocess.call(['git', 'remote', 'rm', remote])

def get_students():
    data = pd.read_csv('./class_2_remote_bis.csv')
    students = {row.Student_Name: row.Remote 
            for i, row in data.iterrows()
            if not pd.isnull(row.Remote) }
    return students
        


def main():
    data = pd.read_csv('./class_2_remote_bis.csv')
    for i, row in data.iterrows():
        remote = row.Remote
        name   = row.Student_Name
        if not pd.isnull(remote):
            try:
                git_remote_add_info232(remote)
                print("Add {} : {}".format(remote, name))
            except AssertionError:
                pass

    # clear_remote()


if __name__ == '__main__':
    main()