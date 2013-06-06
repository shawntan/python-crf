import unittest
import numpy as np
import crf
class TestCRF(unittest.TestCase):
	def setUp(self):
		self.matrix = 0.001 + np.random.poisson(lam=1.5, size=(3,3)).astype(np.float)
		self.vector = 0.001 + np.random.poisson(lam=1.5, size=(3,)).astype(np.float)
		self.M = 0.001 + np.random.poisson(lam=1.5, size=(3,3,3)).astype(np.float)
		self.crf = crf.CRF([],[])
	def test_log_dot_mv(self):
		self.assertTrue(
			(np.around(np.exp(
				crf.log_dot_mv(
					np.log(self.matrix),
					np.log(self.vector)
					)
				),10) == np.around(np.dot(self.matrix,self.vector),10)).all()
			)

	def test_log_dot_vm(self):
		self.assertTrue(
			(np.around(np.exp(
				crf.log_dot_vm(
					np.log(self.vector),
					np.log(self.matrix)
					)
				),10) == np.around(np.dot(self.vector,self.matrix),10)).all()
			)
	def test_forward(self):
		M = self.M/self.M.sum(axis=2).reshape(self.M.shape[:-1]+(1,))
		res = np.around(np.exp(self.crf.forward(np.log(M))[0]).sum(axis=1),10)
		res_true = np.around(np.ones(M.shape[0]),10)
		self.assertTrue((res == res_true).all())
	
	def test_backward(self):
		M = self.M/self.M.sum(axis=1).reshape((3,1,3))
		res = np.around(np.exp(self.crf.backward(np.log(M))[0]).sum(axis=1),10)
		res_true = np.around(np.ones(M.shape[0]),10)
		self.assertTrue((res == res_true).all())

		



if __name__ == '__main__':
	unittest.main()

