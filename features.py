import re
from collections import defaultdict
def listify(gen):
	"Convert a generator into a function which returns a list"
	def patched(*args, **kwargs):
		return list(gen(*args, **kwargs))
	return patched

def fit_dataset(filename):
	labels = set()
	obsrvs = set()
	word_sets = defaultdict(set)
	
	sents_words  = []
	sents_labels = []

	for line in open(filename,'r'):
		sent_words  = []
		sent_labels = []
		for token in line.strip().split():
			word,label= token.split('/')
			word = word.lower()
			labels.add(label)
			obsrvs.add(word)
			word_sets[label].add(word)
			sent_words.append(word)
			sent_labels.append(label)
		sents_words.append(sent_words)
		sents_labels.append(sent_labels)
	return (labels,obsrvs,word_sets,sents_words,sents_labels)

@listify
def set_membership(word_sets):
	for tag in word_sets:
		def fun(yp,y,x_v,i):

			if i < len(x_v):
				if x_v[i].lower() in word_sets[tag]:
					return 1
			else: return 0
		yield fun

@listify
def match_regex(*regexps):
	for regexp in regexps:
		p = re.compile(regexp)
		def fun(yp,y,x_v,i):
			if i < len(x_v) and p.match(x_v[i]): return 1
			else: return 0
		yield fun

regex_functions= ['^[^0-9a-zA-Z]+$','^[A-Z\.]+$','^[0-9\.]+$']
