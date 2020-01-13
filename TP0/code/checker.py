truth = [1, 420, 37571135275072476912863460334485873127534, 7.3982790000000005, 2056.130, 1, 4, 3, 6, 2.97]

def check(answer, question):
	print(answer)
	print(truth[question-1])
	if abs(answer - truth[question-1])<0.0001:
		get_ipython().run_cell_magic(u'HTML', u'', u'<div style="background:#00FF00">CORRECT<br>:-)</div>')
		return 1
	else:
		get_ipython().run_cell_magic(u'HTML', u'', u'<div style="background:#FF0000">BOOOOH<br>:-(</div>')
		return 0