import unittest
from main import func1

class SomeTest(unittest.TestCase):
	def setUp(self):
		super(SomeTest, self).setUp()
		self.mock_data = func1(2);
	
	def test(self):
		self.assertEqual(self.mock_data,4)
		
	def tearDown(self):
		super(SomeTest, self).tearDown()
		self.mock_data = []

if __name__=="__main__":
	unittest.main()