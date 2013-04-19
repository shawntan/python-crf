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
	def __init__(self,feature_functions,labels,sigma=2):
		self.ft_fun = feature_functions
		self.theta  = np.zeros(len(self.ft_fun))
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

	def forward(self,M):
		alphas = np.NINF*np.ones((M.shape[0],M.shape[1]))
		alpha  = alphas[0]
		alpha[self.label_id[START]] = 0
		for i in range(M.shape[0]-1):
			alpha = alphas[i+1] = log_dot_vm(alpha,M[i])
		alpha = log_dot_vm(alpha,M[-1])
		return (alphas,alpha)

	def backward(self,M):
		betas = np.NINF*np.ones((M.shape[0],M.shape[1]))
		beta  = betas[-1]
		beta[self.label_id[END]] = 0
		for i in reversed(range(M.shape[0]-1)):
			beta = betas[i] = log_dot_mv(M[i+1],beta)
		beta = log_dot_mv(M[0],beta)
		return (betas,beta)

	def neg_likelihood_and_deriv(self,x_vec,y_vec,theta,debug=False):
		length = len(x_vec)
		"""
		all_features:	len(x_vec) + 1 x Y x Y x K
		M:				len(x_vec) + 1 x Y x Y
		alphas:			len(x_vec) + 1 x Y
		betas:			len(x_vec) + 1 x Y
		log_probs:		len(x_vec) + 1 x Y x Y  (Y is the size of the state space)
		`unnormalised` value here is alpha * M * beta, an unnormalised probability
		"""
		y_vec           = [START] + y_vec + [END]
		yp_vec_ids      = [ self.label_id[yp] for yp in y_vec[:-1] ]
		y_vec_ids       = [ self.label_id[y]  for y  in y_vec[1:]  ]
		all_features    = self.all_features(x_vec)
		log_M           = np.dot(all_features,theta)
		log_alphas,last = self.forward(log_M)
		log_betas, zero = self.backward(log_M)
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
		#print log_Z
		"""
		Find the expected value of f_k over all transitions
				 and emperical values
		(numpy makes it so easy, only if you do it right)
		"""

		exp_features = np.sum( np.exp(log_probs) * all_features, axis= (0,1,2) )
		emp_features = np.sum( all_features[range(length+1),yp_vec_ids,y_vec_ids], axis = 0 )
		if debug:
			print "ExpFeatures:",exp_features
			print "EmpFeatures:",emp_features
		return (
				- (np.sum(log_M[range(length+1),yp_vec_ids,y_vec_ids]) - log_Z - self.regulariser(theta)), 
				- (emp_features - exp_features - self.regulariser_deriv(theta))
			)

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

	l = lambda theta: crf.neg_likelihood_and_deriv(x_vec,y_vec,theta)
	#crf.theta = optimize.fmin_bfgs(l, crf.theta, maxiter=100)
	crf.theta = optimize.fmin_l_bfgs_b(l, crf.theta)
	print crf.theta
	print crf.neg_likelihood_and_deriv(x_vec,y_vec,crf.theta[0],debug=True)
