"""
TODO:
	x modify function so that it returns (func, gradient)
	x implement regularisation
	- implement viterbi
"""

import numpy as np
from scipy import misc,optimize

START = '|-'
END   = '-|'

def log_dot_vm(loga,logM):
	return misc.logsumexp(loga.reshape(loga.shape+(1,))+logM,axis=0)
def log_dot_mv(logM,logb):
	return misc.logsumexp(logM+logb.reshape((1,)+logb.shape),axis=1)

class CRF:
	def __init__(self,feature_functions,labels,sigma=100):
		self.ft_fun = feature_functions
		self.theta  = np.random.randn(len(self.ft_fun))
		self.labels = [START] + labels + [ END ]
		self.label_id  = { l:i for i,l in enumerate(self.labels) }
		v = sigma ** 2
		v2 = v * 2
		self.regulariser = lambda w: np.sum(w ** 2) / v2
		self.regulariser_deriv = lambda w:np.sum(w) / v

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
		observations = [ self.all_features(x_vec) for x_vec in x_vecs ]
		labels = len(y_vecs)*[None]
	
		for i in range(len(y_vecs)):
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
	def fun(self,i,yp,y,k):
		#print (i,yp,y,k)
		fun = self.ft_fun[k]
		return fun(self.labels[yp],self.labels[y],x_vec,i)
	
	def predict(self,x_vec, debug=False):
		# small overhead, no copying is done
		"""
		all_features:	len(x_vec+1) x Y' x Y x K
		log_potential:	len(x_vec+1) x Y' x Y
		argmaxes:		len(x_vec+1) x Y'
		"""
		all_features  = self.all_features(x_vec)
		log_potential = np.dot(all_features,self.theta)
		return [ self.labels[i] for i in self._predict(log_potential,len(x_vec),len(self.labels)) ]
	
	def _predict(self,log_potential,N,K,debug=False):
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
	
if __name__ == "__main__":
	labels = ['A','B','C']
	obsrvs = ['a','b','c','d','e','f']
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
	x_vec = ["a","b","c","d","e","f"]
	y_vec = ["A","B","C","A","B","C"]

	vectorised_x_vecs,vectorised_y_vecs = crf.create_vector_list([x_vec],[y_vec])
	l = lambda theta: crf.neg_likelihood_and_deriv(vectorised_x_vecs,vectorised_y_vecs,theta)
	#crf.theta = optimize.fmin_bfgs(l, crf.theta, maxiter=100)
	#theta,_,_ = optimize.fmin_l_bfgs_b(l, crf.theta)
	theta = crf.theta
	for _ in range(10000):
		value, gradient = l(theta)
		print value
		theta = theta - 0.01*gradient
	crf.theta = theta
	print theta
	print "Minimized...."
	print crf.neg_likelihood_and_deriv(vectorised_x_vecs,vectorised_y_vecs,crf.theta)
	print
	print crf.predict(x_vec)
