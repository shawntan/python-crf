from crf import CRF
import features
import re, sys
import cloudpickle
training_file = sys.argv[1]

if __name__ == '__main__':
	labels,obsrvs,word_sets,word_data,label_data = features.fit_dataset(training_file)
	crf = CRF(
			labels=list(labels),
			feature_functions = features.set_membership(labels,*word_sets.values()) +
								features.match_regex(labels,
									'^[^0-9a-zA-Z\-]+$',
									'^[^0-9\-]+$',
									'^[A-Z]+$',
									'^-?[1-9][0-9]*\.[0-9]+$',
									'^[1-9][0-9\.]+[a-z]+$',
									'^[0-9]+$',
									'^[A-Z][a-z]+$',
									'^([A-Z][a-z]*)+$',
									'^[^aeiouAEIOU]+$'
								))# + [
								#	lambda yp,y,x_v,i,_y=_y,_x=_x:
								#		1 if i < len(x_v) and y==_y and x_v[i].lower() ==_x else 0
								#	for _y in labels
								#	for _x in obsrvs
								#])
	crf.train(word_data[:-5],label_data[:-5])
	cloudpickle.dump(crf,open(sys.argv[2],'wb'))
	for i in range(-5,0):
		print word_data[i]
		print crf.predict(word_data[i])
		print label_data[i]
