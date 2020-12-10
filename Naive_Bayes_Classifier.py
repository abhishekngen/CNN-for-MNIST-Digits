import _pickle as cp 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import math
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
iris = load_iris()
X, y = iris['data'], iris['target']

class NBC:
    def __init__(self, feature_types, num_classes):
        self.feature_types = feature_types
        self.num_classes = num_classes
        self.D = len(feature_types)
        self.thetaparams = []
        self.piparams = []
        self.ncvals = []

    def fit(self, Xtrain, ytrain):
        N_train = Xtrain.shape[0]
        for c in range(0, self.num_classes):
                paramcolumn = []
                for j in range(0, self.D):
                    mean = 0
                    var = 0
                    count = 0
                    for x in range(0, N_train):
                        if(ytrain[x] == c):
                            mean += Xtrain[:,j][x]
                            count += 1
                    if(count != 0):
                        mean = mean/count
                    for x in range(0, N_train):
                        if(ytrain[x] == c):
                            var += (mean - Xtrain[:,j][x])**2
                    if(count != 0):
                        var = var/count
                    else:
                        var = math.e**(-6)
                    paramcolumn.append((mean, var))
                self.piparams.append(count/N_train)
                self.ncvals.append(count)
                self.thetaparams.append(paramcolumn)
    
    def predict(self, Xtest):
        ypreds = []
        for i in range(0, Xtest.shape[0]):
            highProb = -math.inf
            highclass = 0
            currentProb = 0
            for c in range(0, self.num_classes):
                for j in range(0, len(Xtest[i])):
                        currentProbCalc = 0
                        theta = self.thetaparams[c][j]
                        mu = theta[0]
                        var = theta[1]
                        var = var + 1e-6
                        exponent = math.exp(-((Xtest[i][j] - mu) ** 2 / (2 * var)))
                        
                        if(math.sqrt(var) == 0):
                            currentProbCalc = 0
                        else:
                            currentProbCalc += (1 / (math.sqrt(2 * math.pi) * math.sqrt(var))) * exponent
                        # currentProbCalc = math.log(currentProbCalc) + self.ncvals[c]*math.log(self.piparams[c])
                        if(currentProbCalc<=0):
                            currentProbCalc = 0
                        else:
                            currentProbCalc = math.log(currentProbCalc)
                        currentProb += currentProbCalc
                if(currentProbCalc > highProb):
                    highProb = currentProbCalc
                    highclass = c
            ypreds.append(highclass)
        return ypreds

        
nbc = NBC(feature_types=['r', 'r', 'r', 'r'], num_classes=3)
model = LogisticRegression(max_iter=400)

nbcTestErrors = np.empty((0, 10), float)
lrTestErrors = np.empty((0, 10), float)
nbcAvgTestErrors = []
lrAvgTestErrors = []
trainingSizes = []
N, D = X.shape
Ntrain = int(0.8 * N)
for k in range(1, 11):
    NtrainK = 0.1*k*Ntrain
    trainingSizes.append(NtrainK)
for i in range(0, 200):
    shuffler = np.random.permutation(N) 
    XtrainTotal = X[shuffler[:Ntrain]] 
    ytrainTotal = y[shuffler[:Ntrain]]
    Xtest = X[shuffler[Ntrain:]]
    ytest = y[shuffler[Ntrain:]]
    nbcTempTestErrors = []
    lrTempTestErrors = []
    for k in range(1, 11):
        NtrainK = int(0.1*k*Ntrain)
        Xtrain = XtrainTotal[:NtrainK]
        ytrain = ytrainTotal[:NtrainK]
        nbc = NBC(feature_types=['r', 'r', 'r', 'r'], num_classes=3)
        nbc.fit(Xtrain, ytrain)
        yhatNbc = nbc.predict(Xtest)
        test_error = np.mean(yhatNbc != ytest)
        nbcTempTestErrors.append(test_error)
        model.fit(Xtrain, ytrain)
        yhatLr = model.predict(Xtest)
        test_error = np.mean(yhatLr != ytest)
        lrTempTestErrors.append(test_error)
    nbcTestErrors = np.append(nbcTestErrors, np.array([nbcTempTestErrors]), axis=0)
    lrTestErrors = np.append(lrTestErrors, np.array([lrTempTestErrors]), axis=0)
for col in range(0, 10):
    nbcavg = np.mean(nbcTestErrors[:,col])
    lravg = np.mean(lrTestErrors[:, col])
    nbcAvgTestErrors.append(nbcavg)
    lrAvgTestErrors.append(lravg)
plt.plot(trainingSizes, nbcAvgTestErrors, label = "NBC Error")  
plt.plot(trainingSizes, lrAvgTestErrors, label = "LR Error")
plt.show()