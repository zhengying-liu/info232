truth = [39.6587674942419,1,-1.3145198580879243,3,-0.0001927453473634895]

def check(answer, question):
	print(answer)
	print(truth[question-1])
	if abs(answer - truth[question-1])<0.0001:
		get_ipython().run_cell_magic(u'HTML', u'', u'<div style="background:#00FF00">CORRECT<br>:-)</div>')
		return 1
	else:
		get_ipython().run_cell_magic(u'HTML', u'', u'<div style="background:#FF0000">BOOOOH<br>:-(</div>')
		return 0