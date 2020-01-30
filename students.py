# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

from git_call import git_remote_add_FakeClass

STUDENTS = {
'victor-estrade': 'Estrade Victor',
# 'Didayolo':  'Pavao Adrien',
'herilalaina': 'Rakotoarison Herilalaina',
# 'fake_student':  'No Git Found test',
}


def main():
	for remote, name in STUDENTS.items():
		try:
			git_remote_add_FakeClass(remote)
			print("Add {} : {}".format(remote, name))
		except AssertionError:
			pass

if __name__ == '__main__':
	main()