import numpy as np 
import math
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(X):
    return 1/(1+np.exp(-X))

def sigmoid_derivative(X,z):
    return  X*sigmoid(z)*(1-sigmoid(z))

def build_model(X,output_dim):
    model = {}
    input_dim = X.shape[1]
    model['W'] = np.random.uniform(low=-0.5, high=0.5, size=(input_dim, output_dim))*0.01
    model['b'] = np.zeros((1, output_dim))
    return model

def feed_forward(model, X):
    W, b = model['W'], model['b']

    z = np.dot(X,W) + b
    a = sigmoid(z)

    return z,a

def calculate_loss(A2,Y):

    logprobs = np.multiply(np.log(A2),Y) + np.multiply(np.log(1-A2),1-Y)
    cost = float(np.squeeze(-(1/Y.shape[0])*np.sum(logprobs)))
    return cost

def backprop(X,Y,A,z,model):
    W,b= model['W'],  model['b']
    samples = Y.shape[0]

    dZ = A - Y
    dW = (1/samples) * np.dot(X.T, dZ)
    db = (1/samples) * np.sum(dZ)
    return dW, db


def train(model, X, y, learning_rate):

    previous_loss = float('inf')
    losses = []
    preds = 0
    for i in range(100):
        #feed forward
        z1,a1 = feed_forward(model, X)
        preds = a1
        cost = calculate_loss(a1,y)
        #backpropagation
        dW, db = backprop(X,y,a1,z1,model)
        #updating weights and biases
        model['W'] -= learning_rate * dW
        model['b'] -= learning_rate * db
        if i % 1000 == 0:
            loss = calculate_loss(a1, y)
            losses.append(loss)
            print ("Loss after iteration %i: %f" %(i, loss))
            previous_loss = loss

    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            if(preds[i,j]>=0.5):
                preds[i,j] = 1
            else:
                preds[i,j] = 0

    return model, losses, preds

def main():

    X = [[1,4],[1,5],[2,4],[2,5],[3,1],[3,2],[4,1],[4,2]]
    y = [[0],[0],[0],[0],[1],[1],[1],[1]]

    X = np.asarray(X)
    y = np.asarray(y)

    X_test = [[1,6],[1,9],[2,3],[2,8],[3,4],[3,6],[4,5],[4,6]]
    y_test = [[0],[0],[1],[0],[1],[1],[1],[1]]
   
    nn_input_dim = 2 # input layer dimensionality
    nn_output_dim = 1 # output layer dimensionality 
    learning_rate = 0.08 # learning rate for gradient descent

  
    model = build_model(X,1)
    model, losses, preds = train(model,X, y, learning_rate=learning_rate)
    preds = preds.flatten()

    from sklearn.metrics import accuracy_score
    acct = accuracy_score(y, preds)
    print("Training set accuracy = ",acct)

    z, a = feed_forward(model,X_test)
    a = a.flatten()
    for i in range(len(a)):
        if(a[i]>=0.1):
            a[i]=1
        else:
            a[i]=0
    print(a)
    acctest = accuracy_score(y_test, a)
    print("Test set accuracy = ",acctest)

if __name__ == "__main__":
    main()