import unittest
import main

class SomeTest(unittest.TestCase):
	def setUp(self):
		super(SomeTest, self).setUp()
		self.mock_data = main.sigmoid(0)
	
	def test(self):
		self.assertEqual(self.mock_data,0.5)
		
	def tearDown(self):
		super(SomeTest, self).tearDown()
		self.mock_data = []

if __name__=="__main__":
	unittest.main()