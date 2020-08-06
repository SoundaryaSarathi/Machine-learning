#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 23:07:01 2019

@author: nish03, Sharang Iyer, Kartik Bhide
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
final=[]
df=pd.read_csv('data.csv')      #replace with your file name
X1=df[['d','a']]                #column names
X=(X1/np.max(X1))               #scaling
y1=np.array((df['e']),dtype=float)[:, np.newaxis]
y = y1/(np.max(y1))                #scaling
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
pd.set_option('display.width',None)
np.set_printoptions(threshold=np.inf)

class Neural_Network(object):
    def __init__(self):
        self.inputSize = 2              
        self.outputSize = 1
        self.hiddenSize = 8             #neurons in hidden layer 1
        self.hiddenSize2= 8             #neurons in hidden layer 2
        self.hiddenSize3=8              #neurons in hidden layer 3      (Additional layers can be added with appropriate changes in the code)
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)          #weights
        self.W2 = np.random.randn(self.hiddenSize, self.hiddenSize2)
        self.W3 = np.random.randn(self.hiddenSize2,self.hiddenSize3)
        self.W4 = np.random.randn(self.hiddenSize3,self.outputSize)
        
    def forward(self, X):                           #forward propagation
        self.z = np.dot(X, self.W1)
        self.z2 = self.sigmoid(self.z) 
        self.z3 = np.dot(self.z2, self.W2) 
        self.z4 = self.sigmoid(self.z3)
        self.z5 = np.dot(self.z4,self.W3)
        self.z6 = self.sigmoid(self.z5)
        self.z7 = np.dot(self.z6,self.W4)
        o = self.sigmoid(self.z7)
        return o

    def sigmoid(self, s):                   #activation function (can be changed)
        return 1/(1+np.exp(-s))

    def sigmoidPrime(self, s):              #derivative of activation
        return s * (1 - s)

    def backward(self, X_train, y_train, o):        #backward propagation
        self.o_error = y_train - o
        self.o_delta = self.o_error*self.sigmoidPrime(o) 

        self.z6_error = self.o_delta.dot(self.W4.T)
        self.z6_delta = self.z6_error*self.sigmoidPrime(self.z6)

        self.z4_error = self.z6_delta.dot(self.W3.T) 
        self.z4_delta = self.z4_error*self.sigmoidPrime(self.z4)

        self.z2_error = self.z4_delta.dot(self.W2.T)
        self.z2_delta = self.z4_error*self.sigmoidPrime(self.z2)

        self.W1 += X_train.T.dot(self.z2_delta)*0.0005 
        self.W2 += self.z2.T.dot(self.z4_delta)*0.0005 
        self.W3 += self.z4.T.dot(self.z6_delta)*0.0005
        self.W4 += self.z6.T.dot(self.o_delta)*0.0005
        
    def train(self, X_train, y_train):          #training
        o = self.forward(X_train)
        self.backward(X_train, y_train, o)
        
    def saveWeights(self):
        np.savetxt("w1.txt", self.W1, fmt="%s")
        np.savetxt("w2.txt", self.W2, fmt="%s")
    
    def predict(self):                  #prediction
        print( "Predicted data based on trained weights: ")
        print( "Input (scaled): \n" + str(X_test))
        print( "Output: \n" + str(self.forward(X_test)))



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=98)          #splitting
error=[]
NN = Neural_Network()
for i in range(10000):          #training ; loop according to requirement
      print ("# " + str(i) + "\n")
      print ("Input (scaled): \n" + str(X_train))
      print ("Actual Output: \n" + str(y_train))
      print ("Predicted Output: \n" + str(NN.forward(X_test)))
      print ("Loss: \n" + str(np.mean(np.square(y - NN.forward(X))))) # mean sum squared loss
      print ("\n")
      NN.train(X_train, y_train)
 
NN.saveWeights()
NN.predict()
