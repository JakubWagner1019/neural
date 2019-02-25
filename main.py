import numpy as np
import math

def sigmoid(x):
	return 1/(1+math.e ** (-x))
	
def sigmoid_derivative(x):
	return sigmoid(x)*(1-sigmoid(x))
	
def scale(x,in_min,in_max,out_min,out_max):
	return (out_max-out_min)/(in_max-in_min)*(x-in_min)+out_min
	
def normalize(x,in_min,in_max):
	return scale(x,in_min,in_max,-1,1)

class NeuralNet:
	def __init__(self,input_size,output_size,hidden_size):
		self.input_size = input_size
		self.output_size = output_size
		in_sizes = [input_size,*hidden_size]
		out_sizes = [*hidden_size,output_size]
		self.W = list(np.random.uniform(size = a) for a in zip(in_sizes,out_sizes))
	def learn(self,input,output,learning_rate):
		out = self.run(input)
		E = out - output
		delta = E * learning_rate * sigmoid_derivative(out)
		for i in reversed(range(len(self.W))):
			self.W[i] -= self.A[i].T.dot(delta)
			delta = delta.dot(self.W[i].T) * sigmoid_derivative(self.A[i])
	def run(self,input):
		self.A = list([] for _ in range(len(self.W)+1))
		self.A[0] = input
		for i,W in enumerate(self.W):
			self.A[i+1] = sigmoid(np.dot(self.A[i],W))
		return self.A[-1]

if __name__=="__main__":
	X = np.array([[0,0],[0,1],[1,0],[1,1]])
	Y = np.array([[0],[1],[1],[0]])
	input_size = 2
	output_size = 1
	hidden_size = [3]
	print("input size: {} output size: {} hidden size: {}".format(input_size,output_size,hidden_size))
	a = NeuralNet(input_size,output_size,hidden_size)
	for i in range(50000):
		a.learn(X,Y,0.1)
	print("input: {} output: {}".format([0,0],(a.run([0,0]))))
	print("input: {} output: {}".format([0,1],(a.run([0,1]))))
	print("input: {} output: {}".format([1,0],(a.run([1,0]))))
	print("input: {} output: {}".format([1,1],(a.run([1,1]))))