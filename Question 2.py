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
    model['b'] = np.ones((1, output_dim))
    # model['b'] = np.zeros((1, output_dim))
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
    for i in range(1):
        #feed forward
        z1,a1 = feed_forward(model, X)
        preds = a1
        cost = calculate_loss(a1,y)

        #backpropagation
        dW, db = backprop(X,y,a1,z1,model)

        #updating weights and biases
        model['W'] -= learning_rate * dW
        model['b'] -= learning_rate * db
        
        #Loss calculation
        loss = calculate_loss(a1, y)
        losses.append(loss)
        print ("Loss after iteration %i: %f" %(i+1, loss))
        previous_loss = loss

    pred_new = []
    for pred in preds:
        pred_new.append(np.argmax(pred)+1)

    return model, losses, pred_new

def main():

    X = [[1,1],[1,2],[2,-1],[2,0],[-1,2],[-2,1],[-1,-1],[-2,-2]]
    y = [[1],[1],[2],[2],[3],[3],[4],[4]]

    y_new = []
    template = [0,0,0,0]
    for i in range(len(y)):
        temp = template.copy()
        temp[y[i][0]-1] = temp[y[i][0]-1] + 1
        y_new.append(temp)

    X = np.asarray(X)
    y_new = np.asarray(y_new)

    nn_input_dim = 2 # input layer dimensionality
    nn_output_dim = 4 # output layer dimensionality 
    # learning_rate = 0.09 # learning rate for gradient descent
    learning_rate = 0.009

    #------------Training---------------
    model = build_model(X,nn_output_dim)
    model, losses, preds = train(model,X, y_new, learning_rate=learning_rate)

    from sklearn.metrics import accuracy_score
    print(preds)
    acct = accuracy_score(y, preds)
    print("Training set accuracy = ",acct)
    print("\n")

    #-------------Testing------------
    X_test = [[4,5],[6,8],[3,-6],[6,-4],[-4,6],[-5,7],[-5,-8],[-4,-9]]
    y_test = [[1],[1],[2],[2],[3],[3],[4],[4]]

    z, a = feed_forward(model,X_test)
    print(a)

    a_new = []
    for each_a in a:
        a_new.append(np.argmax(each_a)+1)
    print(y_test)
    acctest = accuracy_score(y_test, a_new)
    print("Test set accuracy = ",acctest)

if __name__ == "__main__":
    main()