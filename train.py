from crf import CRF
import features
import re, sys
import cPickle as pickle
training_file = sys.argv[1]

if __name__ == '__main__':
	labels,obsrvs,word_sets,word_data,label_data = features.fit_dataset(training_file)
	crf = CRF(
			labels=list(labels),
			feature_functions = features.set_membership(word_sets) +
								features.match_regex('^[^0-9a-zA-Z]+$','^[A-Z\.]+$','^[0-9\.]+$')
		)
	crf.train(word_data,label_data)
	pickle.dump(crf,open(sys.argv[2],'wb'))

