import sys
import numpy as np
import matplotlib as mtl
# np.random.seed(0)
import nnfs
import math
from nnfs.datasets import spiral_data

nnfs.init()

inputs  = [1, 2, 3]
weights = [0.2, 0.8, -0.5]
bias = 2

output = inputs[0]* weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias
# print(output)


inputs  = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2

output = inputs[0]* weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + inputs[3]*weights[3] + bias
# print(output)

# inputs  = [1, 2, 3, 2.5]
# weights1 = [0.2, 0.8, -0.5, 1.0]
# weights2 = [0.5, -0.91, 0.26, -0.5]
# weights3 = [-0.26, -0.27, 0.17, 0.87]

# bias1 = 2
# bias2 = 3
# bias3 = 0.5

# output = [inputs[0]* weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
# 		  inputs[0]* weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
# 		  inputs[0]* weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3]

# print(output)


inputs  = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]


# layer_outputs = []

# for neuron_weights, neuron_bias in zip(weights, biases):
# 	neuron_output = 0
# 	for n_input, weight in zip(inputs, neuron_weights):
# 		neuron_output += n_input*weight
# 	neuron_output += neuron_bias
# 	layer_outputs.append(neuron_output)


# print(layer_outputs)


# inputs  = [[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]] # 3*4
# weights = [[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]] # 3*4
# biases = [2, 3, 0.5]

# weights2 = [[0.1,-0.14, 0.5], [-0.5, 0.12, -0.33], [-0.44, 0.73, -0.13]]
# biases2 = [-1, 2, -0.5]



# layer1_output = np.dot(inputs, np.array(weights).T) + biases
# layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2
# print(layer2_output)

X  = [
			[1, 2, 3, 2.5], 
			[2.0, 5.0, -1.0, 2.0], 
			[-1.5, 2.7, 3.3, -0.8]
		] 

X, y = spiral_data(100, 3)

class Layer_Dense(object):
	"""docstring for Layer_Dense"""
	def __init__(self, n_inputs, n_neurons):
		self.weights = np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons))
	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU(object):
	"""docstring for Activation_ReLU"""
	def forward(self, inputs):
		self.output = np.maximum(0, inputs)
		


# layer1 = Layer_Dense(4,5)
# layer2 = Layer_Dense(5, 2)

# layer1.forward(X)
# print(layer1.output)
# print()
# layer2.forward(layer1.output)
# print(layer2.output)
		

# print(0.10*np.random.randn(4,3))


# layer1 = Layer_Dense(2,5)
# acrivation1 =  Activation_ReLU()
# layer1.forward(X)


# acrivation1.forward(layer1.output)
# print(acrivation1.output)
		
layer2_output = [4.8, 1.21, 2.385]

E = math.e

exp_values = [E**X for X in layer2_output]
exp_values = np.exp(layer2_output)
# print(exp_values)

norm_base = sum(exp_values)
# print(norm_base)

norm_values = [value/norm_base for value in exp_values]
norm_values = exp_values / np.sum(exp_values)
# print(norm_values)
# print(sum(norm_values))

layer_output = [[4.8, 1.21, 2.385], [8.9, 1.81, 0.2], [1.41, 1.051, 0.026]]
exp_values = np.exp(layer_output)
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

# print(exp_values)



X, y = spiral_data(100, 3)

class Layer_Dense(object):
	"""docstring for Layer_Dense"""
	def __init__(self, n_inputs, n_neurons):
		self.weights = np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons))
	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU(object):
	"""docstring for Activation_ReLU"""
	def forward(self, inputs):
		self.output = np.maximum(0, inputs)

class Activation_Softmax:
	def forward(self, inputs):
		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
		probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
		self.output = probabilities


X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2,3)
acrivation1 = Activation_ReLU()

dense2 = Layer_Dense(3,3)
acrivation2 = Activation_Softmax()

dense1.forward(X)
acrivation1.forward(dense1.output)

dense2.forward(acrivation1.output)
acrivation2.forward(dense2.output)

print(acrivation2.output[:5])