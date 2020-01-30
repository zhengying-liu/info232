
import os
from subprocess import call

all_files = os.listdir('./')
all_dirs = filter(os.path.isdir, all_files)
TP = list(filter(lambda fname: fname.startswith("PT"), all_files))

for tp in TP:
	cmd = ['python', '-m', 'compileall', f'{tp}/questions.py', '-b']
	print(" ".join(cmd))
	call(cmd)
