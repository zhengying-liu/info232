truth = [151165159337304028706875121056815897906383634117426, 9410.666666666668, 246.5, 44.125, 39.53366570723684, 36.73026713679486, -2, 1.5, 193124410872265632959066965581717890173943794252644069045813220475263627890369816715618, 0.1]

def check(answer, question):
	#print(answer)
	#print(truth[question-1])
	if abs(answer - truth[question-1])<0.0001:
		get_ipython().run_cell_magic(u'HTML', u'', u'<div style="background:#00FF00">CORRECT<br>:-)</div>')
		return 1
	else:
		get_ipython().run_cell_magic(u'HTML', u'', u'<div style="background:#FF0000">BOOOOH<br>:-(</div>')
		return 0