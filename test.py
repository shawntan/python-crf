import unittest
import numpy as np
import crf
class TestCRF(unittest.TestCase):
	def setUp(self):
		self.matrix = 1 + np.random.poisson(lam=1.5, size=(3,3)).astype(np.float)
		self.vector = 1 + np.random.poisson(lam=1.5, size=(3,))
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

if __name__ == '__main__':
	unittest.main()

