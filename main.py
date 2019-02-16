import numpy as np
import math

def sigmoid(x):
	return 1/(1+math.e ** (-x))
	
def sigmoid_derivative(x):
	return sigmoid(x)*(1-sigmoid(x))
	
class NeuralNet:
	def __init__(self,input_size,hidden_size,output_size):
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.W1 = np.random.uniform(size = (self.input_size,self.hidden_size))
		self.W2 = np.random.uniform(size = (self.hidden_size,self.output_size))
	def learn(self,input,output,learning_rate):
		A1 = sigmoid(np.dot(input,self.W1))
		yhat = sigmoid(np.dot(A1,self.W2))
		E = yhat - output
		delta1 = E * learning_rate * sigmoid_derivative(yhat)
		self.W2 -= A1.T.dot(delta1)
		delta2 = delta1.dot(self.W2.T) * sigmoid_derivative(A1)
		self.W1 -= input.T.dot(delta2)
	def run(self,input):
		A1 = sigmoid(np.dot(input,self.W1))
		return sigmoid(np.dot(A1,self.W2))
	
if __name__=="__main__":
	a = NeuralNet(2,3,1)
	X = np.array([[0,0],[0,1],[1,0],[1,1]])
	Y = np.array([[1],[0],[1],[0]])
	for i in range(50000):
		a.learn(X,Y,0.1)
	print("input: {} output: {}".format([0,0],*(a.run([0,0]))))
	print("input: {} output: {}".format([0,1],*(a.run([0,1]))))
	print("input: {} output: {}".format([1,0],*(a.run([1,0]))))
	print("input: {} output: {}".format([1,1],*(a.run([1,1]))))