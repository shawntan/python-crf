import numpy as np
from scipy import misc 

def logdotexp_vec_mat(loga, logM):
	return misc.logsumexp(loga.reshape(loga.shape + (1,))+logM,axis=0)

def logdotexp_mat_vec(logM, logb):
	old = np.array([misc.logsumexp(x + logb) for x in logM], copy=False)
	new = misc.logsumexp(logM+logb.reshape((1,)+logb.shape),axis=1)
	print "N:",new
	print "O:",old
	return old

def logalphas(Mlist):
	logalpha = Mlist[0][0] # alpha(1)
	logalphas = [logalpha]
	for logM in Mlist[1:]:
		logalpha = logdotexp_vec_mat(logalpha, logM)
		logalphas.append(logalpha)
	return logalphas

def logbetas(Mlist):
	logbeta = Mlist[-1][:,2]
	logbetas = [logbeta]
	for logM in Mlist[-2::-1]: # reverse
		logbeta = logdotexp_mat_vec(logM, logbeta)
		logbetas.append(logbeta)
	return logbetas[::-1]
M = [
	np.log(np.array(
	[[0.2,0.2,0.6],
	 [0.2,0.6,0.2],
	 [0.7,0.0,0.3]])),
	np.log(np.array(
	[[0.0,0.0,0.0],
	 [0.2,0.6,0.2],
	 [0.7,0.1,0.2]])),
	np.log(np.array(
	[[0.1,0.2,0.7],
	 [0.2,0.6,0.2],
	 [0.7,0.1,0.2]])),
	]
alphas = np.exp(np.array(logalphas(M)))
betas  = np.exp(np.array(logbetas(M)))
print "Alpha:"
print alphas
print "Beta:"
print betas

