from crf import *
from collections import defaultdict
import re
word_data = []
label_data = []
all_labels = set()
word_sets = defaultdict(set)
obsrvs = set()
for line in open('sample.txt'):
	words,labels = [],[]
	for token in line.strip().split():
		word,label= token.split('/')
		all_labels.add(label)
		word_sets[label].add(word.lower())
		obsrvs.add(word.lower)
		words.append(word)
		labels.append(label)

	word_data.append(words)
	label_data.append(labels)
if __name__ == "__main__":
	labels = list(all_labels)
	lbls   = [START] + labels +  [END]
	transition_functions = [
			lambda yp,y,x_v,i,_yp=_yp,_y=_y: 1 if yp==_yp and y==_y else 0
				for _yp in lbls[:-1] for _y  in lbls[1:]]
	def set_membership(tag):
		def fun(yp,y,x_v,i):
			if i < len(x_v) and x_v[i].lower() in word_sets[tag]:
				return 1
			else:
				return 0
		return fun
	observation_functions = [set_membership(t) for t in word_sets ]
	misc_functions = [
			lambda yp,y,x_v,i: 1 if i < len(x_v) and re.match('^[^0-9a-zA-Z]+$',x_v[i]) else 0,
			lambda yp,y,x_v,i: 1 if i < len(x_v) and re.match('^[A-Z\.]+$',x_v[i]) else 0,
			lambda yp,y,x_v,i: 1 if i < len(x_v) and re.match('^[0-9\.]+$',x_v[i]) else 0 
		]
	tagval_functions = [
			lambda yp,y,x_v,i,_y=_y,_x=_x: 1 if i < len(x_v) and y==_y and x_v[i].lower() ==_x else 0
				for _y in labels
				for _x in obsrvs]
	crf = CRF( labels = labels,
			   feature_functions = transition_functions +  tagval_functions + observation_functions + misc_functions )
	vectorised_x_vecs,vectorised_y_vecs = crf.create_vector_list(word_data,label_data)
	l = lambda theta: crf.neg_likelihood_and_deriv(vectorised_x_vecs,vectorised_y_vecs,theta)
	#crf.theta = optimize.fmin_bfgs(l, crf.theta, maxiter=100)
	print "Minimizing..."
	def print_value(theta):
		print crf.neg_likelihood_and_deriv(vectorised_x_vecs,vectorised_y_vecs,theta)

	#val = optimize.fmin_l_bfgs_b(l, crf.theta)
	#print val
	#theta,_,_  = val
	theta = crf.theta
	for _ in range(10000):
		value, gradient = l(theta)
		print value
		theta = theta - 0.1*gradient
	crf.theta = theta
	print crf.neg_likelihood_and_deriv(vectorised_x_vecs,vectorised_y_vecs,crf.theta)
	print
	print
	for x_vec in word_data[-5:]:
		print x_vec
		print crf.predict(x_vec)
