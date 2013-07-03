"""
TODO:
	x modify function so that it returns (func, gradient)
	x implement regularisation
	- implement viterbi
"""
import marshal
import numpy as np
from scipy import misc,optimize

START = '|-'
END   = '-|'

def log_dot_vm(loga,logM):
	return misc.logsumexp(loga.reshape(loga.shape+(1,))+logM,axis=0)
def log_dot_mv(logM,logb):
	return misc.logsumexp(logM+logb.reshape((1,)+logb.shape),axis=1)

class CRF:
	def __init__(self,feature_functions,labels,sigma=10,transition_feature=True):
		self.ft_fun = feature_functions
		
		self.labels = [START] + labels + [ END ]
		if transition_feature:
			self.ft_fun = self.ft_fun + Transitions.functions(self.labels[1:],self.labels[:-1])
		self.theta  = np.random.randn(len(self.ft_fun))

		self.label_id  = { l:i for i,l in enumerate(self.labels) }
		self.v = sigma ** 2
		self.v2 = self.v * 2

	def regulariser(self,w):
		return np.sum(w ** 2) /self.v2
	def regulariser_deriv(self,w):
		return np.sum(w) / self.v

	def all_features(self,x_vec):
		"""
		Axes:
		0 - T or time or sequence index
		1 - y' or previous label
		2 - y  or current  label
		3 - f(y',y,x_vec,i) for i s
		"""
		result = np.zeros((len(x_vec)+1,len(self.labels),len(self.labels),len(self.ft_fun)))
		for i in range(len(x_vec)+1):
			for j,yp in enumerate(self.labels):
				for k,y in enumerate(self.labels):
					for l,f in enumerate(self.ft_fun):
						result[i,j,k,l] = f(yp,y,x_vec,i)
		return result

	def forward(self,M,start=0):
		alphas = np.NINF*np.ones((M.shape[0],M.shape[1]))
		alpha  = alphas[0]
		alpha[start] = 0
		for i in range(M.shape[0]-1):
			alpha = alphas[i+1] = log_dot_vm(alpha,M[i])
		alpha = log_dot_vm(alpha,M[-1])
		return (alphas,alpha)

	def backward(self,M,end=-1):
		#betas = np.NINF*np.ones((M.shape[0],M.shape[1]))
		betas = np.zeros((M.shape[0],M.shape[1]))
		beta  = betas[-1]
		beta[end] = 0
		for i in reversed(range(M.shape[0]-1)):
			beta = betas[i] = log_dot_mv(M[i+1],beta)
		beta = log_dot_mv(M[0],beta)
		return (betas,beta)

	def create_vector_list(self,x_vecs,y_vecs):
		print len(x_vecs)
		observations = [ self.all_features(x_vec) for x_vec in x_vecs ]
		labels = len(y_vecs)*[None]
	
		for i in range(len(y_vecs)):
			assert(len(y_vecs[i]) == len(x_vecs[i]))
			y_vecs[i].insert(0,START)
			y_vecs[i].append(END)
			labels[i] = np.array([ self.label_id[y] for y in y_vecs[i] ],copy=False,dtype=np.int)
		
		return (observations,labels)

	def neg_likelihood_and_deriv(self,x_vec_list,y_vec_list,theta,debug=False):
		likelihood = 0
		derivative = np.zeros(len(self.theta))
		for x_vec,y_vec in zip(x_vec_list,y_vec_list):
			"""
			all_features:	len(x_vec) + 1 x Y x Y x K
			M:				len(x_vec) + 1 x Y x Y
			alphas:			len(x_vec) + 1 x Y
			betas:			len(x_vec) + 1 x Y
			log_probs:		len(x_vec) + 1 x Y x Y  (Y is the size of the state space)
			`unnormalised` value here is alpha * M * beta, an unnormalised probability
			"""
			all_features    = x_vec
			length 			= x_vec.shape[0]
			#y_vec           = [START] + y_vec + [END]
			yp_vec_ids      = y_vec[:-1]
			y_vec_ids       = y_vec[1:]
			log_M           = np.dot(all_features,theta)
			log_alphas,last = self.forward(log_M,self.label_id[START])
			log_betas, zero = self.backward(log_M,self.label_id[END])
			time,state      = log_alphas.shape
			"""
			Reshaping allows me to do the entire computation of the unormalised
			probabilities in one step, which means its faster, because it's done
			in numpy
			"""
			log_alphas1 = log_alphas.reshape(time,state,1)
			log_betas1  = log_betas.reshape(time,1,state)
			log_Z       = misc.logsumexp(last)
			log_probs   = log_alphas1 + log_M + log_betas1 - log_Z
			log_probs   = log_probs.reshape(log_probs.shape+(1,))
			"""
			Find the expected value of f_k over all transitions
					 and emperical values
			(numpy makes it so easy, only if you do it right)
			"""
			exp_features = np.sum( np.exp(log_probs) * all_features, axis= (0,1,2) )
			emp_features = np.sum( all_features[range(length),yp_vec_ids,y_vec_ids], axis = 0 )

			likelihood += np.sum(log_M[range(length),yp_vec_ids,y_vec_ids]) - log_Z
			derivative += emp_features - exp_features
			if debug:
				print "EmpFeatures:"
				print emp_features
				print "ExpFeatures:"
				print exp_features
		
		return (
			- ( likelihood - self.regulariser(theta)), 
			- ( derivative - self.regulariser_deriv(theta))
		)
	
	def predict(self,x_vec, debug=False):
		# small overhead, no copying is done
		"""
		all_features:	len(x_vec+1) x Y' x Y x K
		log_potential:	len(x_vec+1) x Y' x Y
		argmaxes:		len(x_vec+1) x Y'
		"""
		all_features  = self.all_features(x_vec)
		log_potential = np.dot(all_features,self.theta)
		return [ self.labels[i] for i in self.slow_predict(log_potential,len(x_vec),len(self.labels)) ]
	
	def slow_predict(self,log_potential,N,K,debug=False):
		"""
		Find the most likely assignment to labels given parameters using the
		Viterbi algorithm.
		"""
		g0 = log_potential[0,0]
		g  = log_potential[1:]

		B = np.ones((N,K), dtype=np.int32) * -1
		# compute max-marginals and backtrace matrix
		V = g0
		for t in xrange(1,N):
			U = np.empty(K)
			for y in xrange(K):
				w = V + g[t-1,:,y]
				B[t,y] = b = w.argmax()
				U[y] = w[b]
			V = U
		# extract the best path by brack-tracking
		y = V.argmax()
		trace = []
		for t in reversed(xrange(N)):
			trace.append(y)
			y = B[t, y]
		trace.reverse()
		return trace

	def log_predict(self,log_potential,N,K,debug=False):
		if debug:
			print
			print
			print "Log Potentials:"
			print log_potential
			print
			print
		prev_state    = log_potential[0,self.label_id[START]]
		prev_state_v  = prev_state.reshape((K,1))
		argmaxes      = np.zeros((N,K),dtype=np.int)
		if debug:
			print "T=0"
			print prev_state
			print
		for i in range(1,N):
			curr_state  = prev_state_v + log_potential[i]
			argmaxes[i] = np.nanargmax(curr_state,axis=0)
			prev_state[:]  = curr_state[argmaxes[i],range(K)]
			if debug:
				print
				print "T=%d"%i
				print curr_state
				print prev_state
				print argmaxes[i]
				print
		curr_state = prev_state + log_potential[-1,self.label_id[END]]
		prev_label = np.argmax(curr_state)
		if debug: print prev_label
		result = []
		for i in reversed(range(N)):
			if debug:print result
			result.append(prev_label)
			prev_label = argmaxes[i,prev_label]
		result.reverse()
		return result

	def train(self,x_vecs,y_vecs,debug=False):
		vectorised_x_vecs,vectorised_y_vecs = self.create_vector_list(x_vecs,y_vecs)
		l = lambda theta: self.neg_likelihood_and_deriv(vectorised_x_vecs,vectorised_y_vecs,theta)
		val = optimize.fmin_l_bfgs_b(l,self.theta)
		if debug: print val
		self.theta,_,_  = val
		return self.theta


class FeatureSet(object):
	@classmethod
	def functions(cls,lbls,*arguments):
		def gen():
			for lbl in lbls:
				for arg in arguments:
					if isinstance(arg,tuple):
						yield cls(lbl,*arg)
					else:
						yield cls(lbl,arg)
		return list(gen())
	def __repr__(self):
		return "%s(%s)"%(self.__class__.__name__,self.__dict__)

class Transitions(FeatureSet):
	def __init__(self,curr_lbl,prev_lbl):
		self.prev_label = prev_lbl
		self.label = curr_lbl

	def __call__(self,yp,y,x_v,i):
		if yp==self.prev_label and y==self.label:
			return 1
		else:
			return 0

