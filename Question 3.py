import numpy as np 
import math
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(X):
    return 1/(1+np.exp(-X))

def sigmoid_derivative(X,z):
    return  X*sigmoid(z)*(1-sigmoid(z))

def build_model(X,hidden_nodes,output_dim):
    model = {}
    input_dim = X.shape[1]
    model['W1'] = np.random.uniform(low=-0.5, high=0.5, size=(input_dim, hidden_nodes))
    model['b1'] = np.zeros((1, hidden_nodes))
    model['W2'] = np.random.uniform(low=-0.5, high=0.5, size=(hidden_nodes, output_dim))
    model['b2'] = np.zeros((1, output_dim))
    return model

def feed_forward(model, X):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    z1 = np.dot(X,W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1,W2) + b2
    a2 = sigmoid(z2)

    return z1,a1,z2,a2

def calculate_loss(A2,Y):

    logprobs = np.multiply(np.log(A2),Y) + np.multiply(np.log(1-A2),1-Y)
    cost = float(np.squeeze(-(1/Y.shape[1])*np.sum(logprobs)))
    return cost

def backprop(X,Y,A1,A2,z1,z2,model):
    W1, W2= model['W1'],  model['W2']
    samples = Y.shape[0]

    dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
    dZ2 = sigmoid_derivative(dA2,z2)
    dW2 = (1/samples)*np.dot(A1.T,dZ2)
    db2 = (1/samples)*np.sum(dZ2, axis=0, keepdims = True)

    dZ1 = np.multiply(np.dot(dZ2,W2.T),A1*(1-A1))
    dW1 = (1/samples)*np.dot(X.T,dZ1)
    db1 = (1/samples)*np.sum(dZ1, axis=0, keepdims = True)

    return dW1, dW2, db1, db2


def train(model, X, y, learning_rate):

    previous_loss = float('inf')
    losses = []
    preds = 0
    for i in range(10001):
    	#feed forward
        z1,a1,z2,a2 = feed_forward(model, X)
        preds = a2
        cost = calculate_loss(a2,y)
        #backpropagation
        dW1, dW2, db1, db2 = backprop(X,y,a1,a2,z1,z2,model)
        #updating weights and biases
        model['W1'] -= learning_rate * dW1
        model['b1'] -= learning_rate * db1
        model['W2'] -= learning_rate * dW2
        model['b2'] -= learning_rate * db2
        if i % 1000 == 0:
        	loss = calculate_loss(a2, y)
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

    # X = [[1,4],[1,5],[2,4],[2,5],[3,1],[3,2],[4,1],[4,2]]
    # y = [0,0,0,0,1,1,1,1]
    dfX = pd.read_csv("trainData.csv")
    X = np.asarray(dfX)
    dfY = pd.read_csv("trainLabels.csv")
    y = np.asarray(dfY)

    for i in range(len(y)):
        if(y[i] == 5):
            y[i] = 0
        else:
            y[i] = 1    
    nn_input_dim = 64 # input layer dimensionality
    nn_output_dim = 1 # output layer dimensionality 
    learning_rate = 0.01 # learning rate for gradient descent

    hidden_layer_range = [5,6,7,8,9,10,11,12,13,14,15]
    accuracy_score_train = []
    accuracy_score_test = []
    for layer in hidden_layer_range:
        print("Calculating for layers = ",layer)
        model = build_model(X,layer,1)
        model, losses, preds = train(model,X, y, learning_rate=learning_rate)
        preds = preds.flatten()
        from sklearn.metrics import accuracy_score
        acct = accuracy_score(y, preds)
        print("Training set accuracy for "+str(layer)+" hidden layers = ",acct)
        accuracy_score_train.append(acct)
        

        dfX_test = pd.read_csv("testData.csv")
        X_test = np.asarray(dfX_test)
        dfY_test = pd.read_csv("testLabels.csv")
        y_test = np.asarray(dfY_test)
        z1,a1,z2,a2 = feed_forward(model,X_test)

        for i in range(a2.shape[0]):
            for j in range(a2.shape[1]):
                if(a2[i,j]>=0.5):
                    a2[i,j] = 6
                else:
                    a2[i,j] = 5
        a2 = a2.flatten()
        y_test = y_test.flatten()

        from sklearn.metrics import accuracy_score
        acctest = accuracy_score(y_test, a2)
        print("Test set accuracy for "+str(layer)+" hidden layers = "+str(acctest)+"\n")
        accuracy_score_test.append(acctest)
        

    plot_title = "Number of Hidden Layers vs Training Accuracy"
    plotss(hidden_layer_range, accuracy_score_train, plot_title)

    plot_title = "Number of Hidden Layers vs Test Set Accuracy"
    plotss(hidden_layer_range, accuracy_score_test, plot_title)

def plotss(X,Y,title):
    plt.title(title)
    plt.scatter(X, Y)
    plt.plot(X,Y, color='red') # regression line
    plt.xlabel("Hidden Layers")
    plt.ylabel("Accuracy")
    plt.show()       

if __name__ == "__main__":
    main()