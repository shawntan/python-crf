import re
from collections import defaultdict

alphas = re.compile('^[a-zA-Z]+$')

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

class FeatureSet:
	@classmethod
	def functions(cls,lbls,*arguments):
		def gen():
			for lbl in lbls:
				for arg in arguments:
					print arg
					if isinstance(arg,tuple):
						yield cls(lbl,*arg)
					else:
						yield cls(lbl,arg)
		return list(gen())
	def __repr__(self):
		return "%s(%s)"%(self.__class__.__name__,self.__dict__)

class Membership(FeatureSet):
	def __init__(self,label,word_set):
		self.label = label
		self.word_set = word_set
	def __call__(self,yp,y,x_v,i):
		if i < len(x_v) and y == self.label and (x_v[i].lower() in self.word_set):
			return 1
		else:
			return 0
class FileMembership(Membership):
	def __init__(self,label,filename):
		self.label = label
		self.word_set = set([ line.strip().lower() for line in open(filename,'r') ])

class MatchRegex(FeatureSet):
	def __init__(self,label,regex):
		self.label = label
		self.regex = re.compile(regex)
	def __call__(self,yp,y,x_v,i):
		if i < len(x_v) and y==self.label and self.regex.match(x_v[i]):
			return 1
		else:
			return 0

if __name__ == "__main__":
	val = Membership.functions(['HI','HO'],set(['hi']),set(['ho'])) + MatchRegex.functions(['HI','HO'],'\w+','\d+')
	print val[0]('HO','HO',['hi'],0)
	print val[0]('HO','HI',['hi'],0)
