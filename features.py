import re
from collections import defaultdict

alphas = re.compile('^[a-zA-Z]+$')
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
		try:
			for token in line.strip().split():
				word,label= token.rsplit('/',2)
				if alphas.match(word):
					orig_word = word
					word = word.lower()
					labels.add(label)
					obsrvs.add(word)
					word_sets[label].add(word)
					sent_words.append(orig_word)
					sent_labels.append(label)
				else:
					continue
			sents_words.append(sent_words)
			sents_labels.append(sent_labels)
		except Exception:
			print line
	return (labels,obsrvs,word_sets,sents_words,sents_labels)

@listify
def set_membership(labels,*word_sets):
	for lbl in labels:
		for ws in word_sets:
			def fun(yp,y,x_v,i,lbl=lbl,s=ws):
				if i < len(x_v) and y==lbl and (x_v[i].lower() in s):
					#print lbl, ws,x_v[i]
					return 1
				else: return 0
			yield fun

@listify
def match_regex(labels,*regexps):
	for regexp in regexps:
		p = re.compile(regexp)
		for lbl in labels:
			def fun(yp,y,x_v,i,lbl=lbl,p=p):
				if i < len(x_v) and y==lbl and p.match(x_v[i]):
					return 1
				else: return 0
			yield fun

