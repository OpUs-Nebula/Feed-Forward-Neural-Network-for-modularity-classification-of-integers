# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import copy
import random
##import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
print("Script INIT")

#input_data = np.array([[3, 7, 11], [2, 4, 6], [8, 12, 14], [15, 17, 19]])
#output_labels = np.array([[0], [1], [1], [0]])

#input_data = np.array([[3, 5, 7], [2, 4, 6], [9, 11, 13], [8, 10, 12]])
#output_labels = np.array([[0], [1], [1], [0]])
#If containing multiples of 2, then output is 1, else 0. scale between can be seen as confidence score of beloning to class.


input_data = []
output_labels = []
validation_data = [[],[]]
tmp = [[],[]]

#dict_mod = {0:1,1:0}

max = 300
min = 1
integers = range(min,(max + 1))
randomized_distribution = random.sample(integers,len(integers))

print("Random Distribution:" + str(randomized_distribution))
for int in randomized_distribution:
    div = int%2
    if (len(tmp[div]) < 3):
        tmp[div].append(int)
    else:
        input_data.append(tmp[div])
        tmp[div] = []
        output_labels.append([1 * (div == 0)])

#Feature scaling: a + X' = ((X - X_min) * (b - a)) / (X_max - X_min)) for bounding
#to closed interval [a,b]
#scaled_int = (int - min) / (max - min)
for arr in input_data:
    arr_copy = copy.copy(arr)
    for index,element in enumerate(arr):
        arr[index] = (element - np.amin(arr_copy)) / (np.amax(arr_copy) - np.amin(arr_copy))

size_i = len(input_data)
size_l = len(output_labels)

#to closed interval [a,b]
#scaled_int = (int - min) / (max - min)

input_data = np.array(input_data)
output_labels = np.array(output_labels)

print("Input/Output:")
print(input_data)
print(output_labels)

#print("Input:")
#print(input_data)

#print("Labels:")
#print(output_labels)

#Sigmoid
a = 0.01
def activate(x,deriv=False):
    if(deriv==True):
        return x*(1-x) 
    return 1/(1+np.exp(-x))

##Reciprocal weight matrix to input
##Reciprocal to labels    
synaptic_weights_0 = 2*np.random.random((3,size_i)) - 1
synaptic_weights_1 = 2*np.random.random((size_l,1)) - 1

#print("Weights:")
print (synaptic_weights_0)
print (synaptic_weights_1)

for j in range(60000):
    
    ##Forward propagation, layer 0,1,2
    layer0 = input_data
    layer1 = activate(np.dot(layer0,synaptic_weights_0))
   # print("Layer1:")
    #print(layer1)
    ##print("Synaptic_Weights_1:")
    ##print(synaptic_weights_1)
    layer2 = activate(np.dot(layer1,synaptic_weights_1))
    #print("Layer2:")
    #print(layer2)
    
    ##Calculate Error for layer 2
    layer2_error = output_labels - layer2
    bool = (j%10000 == 0 or j == 0)
    ##Printing layer error for 2
    if bool:
       print("Error:" + str(np.mean(np.abs(layer2_error))))
        #print("Synaptic_Weights_1:")
        #print(synaptic_weights_1)
        #print("Layer1:")
        #print(layer1)
        #print("Layer2:")
        #print(layer2)
    
    ##Gradient
    #if bool:
        #print("Synaptic_weights Transposed:" + str(synaptic_weights_1.T)
    
    layer2_gradient = layer2_error * activate(layer2,deriv=True)
    
    #if bool:
        #print("layer2 Gradient:")
        #print(layer2_gradient)
    ##Error for layer 1
    layer1_error = layer2_gradient.dot(synaptic_weights_1.T)
    
    ##Use error for layer 1 gradient
    layer1_gradient = layer1_error * activate(layer1,deriv=True)
    
    ##print("Layer1_gradient:")
    ##print(layer1_gradient)
    
    ##Updating weights
    synaptic_weights_1 += layer1.T.dot(layer2_gradient)
    synaptic_weights_0 += layer0.T.dot(layer1_gradient)

    ##Exec: python C:\Users\Mbwenga\Documents\PythonScripts\Script.py
    #Printing layer error for 2
    #if bool:
        #print("Synaptic_Weights_1:")
        #print(synaptic_weights_1)
        #print("Synaptic_weights_0:")
        #print(synaptic_weights_0)
        
def predict(input):
    layer0 = input
    layer1 = activate(np.dot(layer0,synaptic_weights_0))
    layer2 = activate(np.dot(layer1,synaptic_weights_1))
    return layer2

#Testing:
##[0,0.5,1] = [100,120,140], or really [k,1/2n,n], k < 1/2n, n >= 1
prediction = predict(np.array([[0,0.5,1]]))
print(prediction)

#Result: 1.0 100% of the times using prewritten array(1 == all numbers are even, 0 == alla numbers are odd), very promising as this
#algorihtm generalizes very well to any array of numbers, if feature scaling is applied beforehand.

    