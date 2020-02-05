truth = [161.359233, 24, 1.9428571428571428, 0.9038461538461539, 2.9230769230769234]

def check(answer, question):
	#print(answer)
	#print(truth[question-1])
	if abs(answer - truth[question-1])<0.0001:
		get_ipython().run_cell_magic(u'HTML', u'', u'<div style="background:#00FF00">CORRECT<br>:-)</div>')
		return 1
	else:
		get_ipython().run_cell_magic(u'HTML', u'', u'<div style="background:#FF0000">BOOOOH<br>:-(</div>')
		return 0
