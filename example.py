from crf import *

word_data = []
label_data = []
all_labels = set()
all_words = set()
for line in open('sample.txt'):
	words,labels = [],[]
	for token in line.strip().split():
		word,label= token.split('/')
		all_labels.add(label)
		all_words.add(word)
		words.append(word)
		labels.append(label)

	word_data.append(words)
	label_data.append(labels)

if __name__ == "__main__":
	labels = list(all_labels)
	obsrvs = list(all_words)
	lbls   = [START] + labels +  [END]
	transition_functions = [
			lambda yp,y,x_v,i,_yp=_yp,_y=_y: 1 if yp==_yp and y==_y else 0
				for _yp in lbls[:-1]
				for _y  in lbls[1:]]
	observation_functions = [
			lambda yp,y,x_v,i,_y=_y,_x=_x: 1 if i < len(x_v) and y==_y and x_v[i]==_x else 0
				for _y in labels
				for _x in obsrvs]
	crf = CRF( labels = labels,
			   feature_functions = transition_functions + observation_functions )

	vectorised_x_vecs = crf.create_vector_list(word_data)
	l = lambda theta: crf.neg_likelihood_and_deriv(vectorised_x_vecs,label_data,theta)
	#crf.theta = optimize.fmin_bfgs(l, crf.theta, maxiter=100)
	print "Minimizing..."
	theta,_,_ = optimize.fmin_l_bfgs_b(l, crf.theta)
	crf.theta = theta
	print crf.neg_likelihood_and_deriv(vectorised_x_vecs,label_data,crf.theta)
	print
	print
	x_vec = word_data[-1]
	print x_vec
	print crf.predict(x_vec)
	print x_vec
