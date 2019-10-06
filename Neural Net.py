#In this programming assignment, you are asked to train a neural net model, which
#classifies data in linear or non-linear distribution.
#Submission: two things should be submitted via our gsubmit system: code which
#trains, evaluates the neural net, and a report explaining your code and experiments

#1. Reading data from .csv files in numpy matrix and split them into k folds.
#2. Initialization of a single-hidden-layer neural network, with adjustable settings such as:
	#a. Number of neurons
	#b. Lamba for regularization
	#c. Learning rate
	#d. Number of epochs
#3. Forward and back propagation for each single data point.
#4. Computing confusion matrix, accuracy, precision, recall and F-1 score.
#5. Visualizing your classification result.


import numpy as np
import os, sys
from numpy import genfromtxt
import math
from matplotlib import pyplot as plt

class NeuralNetwork:
    def __init__(self, NNodes, activate, deltaActivate):
        self.NNodes = NNodes # the number of nodes in the hidden layer
        self.activate = activate # a function used to activate
        self.deltaActivate = deltaActivate # the derivative of activate
    
    def fit(self, X, Y, learningRate, epochs, regLambda):
        """
        This function is used to train the model.
        Parameters
        ----------
        X : numpy matrix
            The matrix containing sample features for training.
        Y : numpy array
            The array containing sample labels for training.
        Returns
        -------
        None
        """

        self.regLambda = regLambda
        self.learningRate = learningRate
        # Initialize your weight matrices first. W
        # (hint: check the sizes of your weight matrices WL and WO first!)
        WL_cols = X.shape[1]+1
        WL_rows = self.NNodes
        self.WL = np.random.rand(WL_rows, WL_cols)
       	
       	WO_cols = self.NNodes + 1
        WO_rows = 1
       	self.WO = np.random.rand(WO_rows, WO_cols)
        
        # For each epoch, do
        for epoch in range(epochs):
            # For each training sample (X[i], Y[i]), do
            for i in range(X.shape[0]):
                # 1. Forward propagate once. Use the function "forward" here!
                YPredict = self.forward(X[i])
                # 2. Backward progate once. Use the function "backpropagate" here!
                self.backpropagate(X[i], Y[i], YPredict)
        pass
        

    def predict(self, X):
        """
        Predicts the labels for each sample in X.
        Parameters
        X : numpy matrix
            The matrix containing sample features for testing.
        Returns
        -------
        YPredict : numpy array
            The predictions of X.
        ----------
        """
        YPredicts = []
        for i in range(X.shape[0]):
        	YPredict = self.forward(X[i])
        	#predict the category with largest value
        	YPredicts.append(YPredict)

        YPredicts = np.array(YPredicts)
        return YPredicts



    def forward(self, X):
        # Perform matrix multiplication and activation twice (one for each layer).
        # (hint: add a bias term before multiplication)
        
       	#Add bias term to input layer and compute activation for hidden layer
        #print(X)
        X = np.hstack((X, 1))
        #print(X)
        Z = np.matmul(self.WL, np.transpose(X))
        #print(Z)
        vActivate = np.vectorize(activate)
        self.A = vActivate(Z)
        self.A = np.hstack((self.A, 1))
        #print("A: ", self.A)
        YPredict = np.matmul(self.WO, self.A)
        #print(YPredict)
        return YPredict
     	
    def backpropagate(self, X, YTrue, YPredict):
        # Compute loss / cost using the getCost function.
        cost = self.getCost(YTrue, YPredict)
        #print("cost: ", cost)        
        # Compute gradient for each layer.
        temp1 = np.multiply(self.WO,self.A)
        #print("temp1: ", temp1)
        temp3 = np.ones(self.A.shape[0])
        #print("temp3: ", temp3)
        temp4 = np.subtract(temp3, self.A)
        #print("temp4: ", temp4)
        temp2 = np.multiply(temp1,temp4)
        #print("temp2: ", temp2)
        X = np.hstack((X, 1))
        temp5 = np.transpose(np.outer(X, temp2))
        temp5 = temp5[0:-1,]
        temp6 = self.learningRate*(YTrue - YPredict)*YPredict*(1 - YPredict)
        #print("temp6: ", temp6)
        delWO = temp6*self.A
        #print("delWO: ", delWO)
        delWL = temp6*temp5
        #print("delWL: ", delWL)
        # Compute weight changes: 
        self.WO = np.subtract(self.WO , delWO)
        self.WL = np.subtract(self.WL, delWL)
        #print("WL : ", self.WL)
        # Update weight matrices.
        pass
        
    def getCost(self, YTrue, YPredict):
        # Compute loss / cost in terms of crossentropy.
        # (hint: your regularization term should appear here)
        error = ((YTrue - YPredict)**2)/2.0
        error = error[0]
        #print("error", error)
        regError = self.regLambda*(np.linalg.norm(self.WO, 2) + np.linalg.norm(self.WL, 2))
        #print("reg error:", regError)
        cost = (error + regError)/2.0
        return cost
        pass

    def hw(self):
    	print("hello world")

def getData(dataSampleFile, dataLabelFile):
    '''
    Returns
    -------
    X : numpy matrix
        Input data samples.
    Y : numpy array
        Input data labels.
    '''
    # TO-DO for this part:
    # Use your preferred method to read the csv files.
    # Write your codes here:
    X = genfromtxt(dataSampleFile, delimiter=',')
    Y = genfromtxt(dataLabelFile, delimiter=',')

    print("size of training data : ", X.shape)
    print("size of testing data : ", Y.shape)
    # Hint: use print(X.shape) to check if your results are valid.
    return X, Y

def splitData(X, Y, K = 5):
    '''
    Returns
    -------
    result : List[[train, test]]
        "train" is a list of indices corresponding to the training samples in the data.
        "test" is a list of indices corresponding to the testing samples in the data.
        For example, if the first list in the result is [[0, 1, 2, 3], [4]], then the 4th
        sample in the data is used for testing while the 0th, 1st, 2nd, and 3rd samples
        are for training.
    '''
    size = X.shape[0]
    indices = range(0, size)
    indices = list(indices)
    # Make sure you shuffle each train list.
    np.random.shuffle(indices)

    train_test_indices = []

    for i in range(K):
    	beg_index = int(i*(size/K))
    	end_index = int((i+1)*(size/K))
    	train_test_indices.append(indices[beg_index:end_index])

    train_test_split = []

    for i in range(K):
    	test = []
    	train = []
    	for j in range(K):
    		if(i == j):
    			test = train_test_indices[i]
    		else:
    			train = train + train_test_indices[i]
    	combo = [train, test]
    	train_test_split.append(combo)
    return train_test_split

    pass


def plotDecisionBoundary(model, X, Y):
    """
    Plot the decision boundary given by model.
    Parameters
    ----------
    model : model, whose parameters are used to plot the decision boundary.
    X : input data
    Y : input labels
    """
    x1_array, x2_array = np.meshgrid(np.arange(-4, 4, 0.01), np.arange(-4, 4, 0.01))
    grid_coordinates = np.c_[x1_array.ravel(), x2_array.ravel()]
    Z = model.predict(grid_coordinates)
    Z = Z.reshape(x1_array.shape)
    plt.contourf(x1_array, x2_array, Z, cmap=plt.cm.bwr)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.bwr)
    plt.show()

def train(XTrain, YTrain, args):
    """
    This function is used for the training phase.
    Parameters
    ----------
    XTrain : numpy matrix
        The matrix containing samples features (not indices) for training.
    YTrain : numpy array
        The array containing labels for training.
    args : List
        The list of parameters to set up the NN model.
    Returns
    -------
    NN : NeuralNetwork object
        This should be the trained NN object.
    """
    # 1. Initializes a network object with given args.
    NN = NeuralNetwork(NNodes, activate, deltaActivate)
    # 2. Train the model with the function "fit".
    # (hint: use the plotDecisionBoundary function to visualize after training) 
    NN.fit(XTrain, YTrain, learningRate, epochs, regLambda)
    plotDecisionBoundary(NN, XTrain, YTrain)
    # 3. Return the model.
    return NN
    
    pass

def test(XTest, model):
    """
    This function is used for the testing phase.
    Parameters
    ----------
    XTest : numpy matrix
        The matrix containing samples features (not indices) for testing.
    model : NeuralNetwork object
        This should be a trained NN model.
    Returns
    -------
    YPredict : numpy array
        The predictions of X.
    """
    YPredict = model.predict(XTest)
    return YPredict
    pass

def getConfusionMatrix(YTrue, YPredict):
    """
    Computes the confusion matrix.
    Parameters
    ----------
    YTrue : numpy array
        This array contains the ground truth.
    YPredict : numpy array
        This array contains the predictions.
    Returns
    CM : numpy matrix
        The confusion matrix.
    """

    numberOfClasses = int(np.amax(YTrue) + 1)
    #print("number of classes = ", numberOfClasses)
    CM = np.zeros((numberOfClasses, numberOfClasses))
    print(CM)
    for i in range(len(YTrue)):
    	trueValue = int(round(YTrue[i]))
    	print("trueValue:", YTrue[i])
    	predictValue = int(YPredict[i][0])
    	print("predicted value:", YPredict[i][0])
    	#print(CM[trueValue, predictValue])
    	CM[trueValue, predictValue] = CM[trueValue, predictValue] + 1
    return CM    

    pass
    
def getPerformanceScores(YTrue, YPredict):
    """
    Computes the accuracy, precision, recall, f1 score.
    Parameters
    ----------
    YTrue : numpy array
        This array contains the ground truth.
    YPredict : numpy array
        This array contains the predictions.
    Returns
    {"CM" : numpy matrix,
    "accuracy" : float,
    "precision" : float,
    "recall" : float,
    "f1" : float}
        This should be a dictionary.
    """

    performance = {}
    CM = getConfusionMatrix(YTrue, YPredict)
    performance["CM"] = CM

    accuracy = np.trace(CM) / len(YTrue)
    performance["accuracy"] = accuracy

    numberOfClasses = int(np.amax(YTrue) + 1)
    precision = 0
    for i in range(1, numberOfClasses):		#don't count 0
    	tp = CM[i, i]
    	colTotals = np.sum(CM, axis=0)
    	iTotal = colTotals[i]
    	precision = precision + (tp / iTotal)
    precision = precision / (numberOfClasses - 1) 	#don't count 0
    performance["precision"] = precision

    recall = 0
    for i in range(1, numberOfClasses):	    #don't count 0
    	tp = CM[i, i]
    	rowTotals = np.sum(CM, axis=1)
    	iTotal = rowTotals[i]
    	recall = recall + (tp / iTotal)
    recall = recall / (numberOfClasses - 1) 	#don't count 0
    performance["recall"] = recall

    performance["f1"] = 2 * precision * recall / (precision + recall)

    return performance
	
    pass

def activate(S):
	return 1/(1+math.exp(-1 * S))

def deltaActivate(S):
	return activate(S) * (1 - activate(S))

dataSampleFile = "/Users/krauser/Documents/BU/AI/P1/Data/DataFor440/dataset1/LinearX.csv"
dataLabelFile = "/Users/krauser/Documents/BU/AI/P1/Data/DataFor440/dataset1/LinearY.csv"
X, Y = getData(dataSampleFile, dataLabelFile)
train_test_split = splitData(X, Y, K = 5)
NNodes = 3
learningRate = 1e-2
epochs = 500
regLambda = 1e-3
args = [NNodes, activate, deltaActivate, learningRate, epochs, regLambda]
for i in range(len(train_test_split)):
	XTrain = X[train_test_split[i][0]]
	YTrain = Y[train_test_split[i][0]]
	model = train(XTrain, YTrain, args)
	YPredict = test(X[train_test_split[i][0]], model)
	perfScore = getPerformanceScores(Y[train_test_split[i][0]], YPredict)
	print(perfScore)
