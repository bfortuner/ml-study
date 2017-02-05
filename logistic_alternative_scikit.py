import math
import numpy as np
import pandas as pd
import sklearn
import pylab

#Source: https://github.com/perborgen/LogisticRegression/blob/master/logistic.py

#min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))


# clean up data
df = pd.read_csv(DATA_DIR+"logistic3.csv", usecols=[0,1,2])
df.head()

df.columns = ["grade1","grade2","label"]
df.head()

df['label'] = df.label.apply(lambda x: float(x.rstrip(';')))

df.head()

# Normalize grades to values between 0 and 1 for more efficient computation
min_max_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1))

# Extract Features + Labels
X = df[["grade1","grade2"]]
X = np.array(X)
X = min_max_scaler.fit_transform(X)
Y = df["label"]
Y = np.array(Y)

print X.shape
print Y.shape
print X[:5]
print Y[:5]

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

# creating testing and training set
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33)
#print X_train,X_test
#print Y_train,Y_test

# train scikit learn model
clf = LogisticRegression()
clf.fit(X_train,Y_train)
print 'score Scikit learn: ', clf.score(X_test,Y_test)

# visualize data, uncomment "show()" to run it
pos = np.where(Y == 1)
neg = np.where(Y == 0)
pylab.scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
pylab.scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
pylab.xlabel('Exam 1 score')
pylab.ylabel('Exam 2 score')
pylab.legend(['Not Admitted', 'Admitted'])
pylab.show()

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

# creating testing and training set
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33)
#print X_train,X_test
#print Y_train,Y_test

# train scikit learn model
clf = LogisticRegression()
clf.fit(X_train,Y_train)
print 'score Scikit learn: ', clf.score(X_test,Y_test)

def Sigmoid(z):
    G_of_Z = float(1.0 / float((1.0 + math.exp(-1.0*z))))
    return G_of_Z

def Hypothesis(theta, x):
    z = 0
    for i in xrange(len(theta)):
        z += x[i]*theta[i]
    return Sigmoid(z)

def Cost_Function(X,Y,theta,m):
    sumOfErrors = 0
    for i in xrange(m):
        xi = X[i]
        hi = Hypothesis(theta,xi)
        if Y[i] == 1:
            error = Y[i] * math.log(hi)
        elif Y[i] == 0:
            error = (1-Y[i]) * math.log(1-hi)
        sumOfErrors += error
    const = -1/m
    J = const * sumOfErrors
    print 'cost is ', J
    return J

def Cost_Function_Derivative(X,Y,theta,j,m,alpha):
    sumErrors = 0
    for i in xrange(m):
        xi = X[i]
        xij = xi[j]
        hi = Hypothesis(theta,X[i])
        error = (hi - Y[i])*xij
        sumErrors += error
    m = len(Y)
    constant = float(alpha)/float(m)
    J = constant * sumErrors
    return J

def Gradient_Descent(X,Y,theta,m,alpha):
    new_theta = []
    constant = alpha/m
    for j in xrange(len(theta)):
        CFDerivative = Cost_Function_Derivative(X,Y,theta,j,m,alpha)
        new_theta_value = theta[j] - CFDerivative
        new_theta.append(new_theta_value)
    return new_theta

def Logistic_Regression(X,Y,alpha,theta,num_iters):
    m = len(Y)
    for x in xrange(num_iters):
        new_theta = Gradient_Descent(X,Y,theta,m,alpha)
        theta = new_theta
        if x % 100 == 0:
            Cost_Function(X,Y,theta,m)
            print 'theta ', theta
            print 'cost is ', Cost_Function(X,Y,theta,m)

def Declare_Winner(theta):
    score = 0
    winner = ""
    scikit_score = clf.score(X_test,Y_test)
    length = len(X_test)
    for i in xrange(length):
        prediction = round(Hypothesis(X_test[i],theta))
        answer = Y_test[i]
    if prediction == answer:
        score += 1
    my_score = float(score) / float(length)
    if my_score > scikit_score:
        print 'You won!'
    elif my_score == scikit_score:
        print 'Its a tie!'
    else:
        print 'Scikit won.. :('
        print 'Your score: ', my_score
    print 'Scikits score: ', scikit_score


# setting variables
initial_theta = [0,0]
alpha = 0.1
iterations = 1000
Declare_Winner(initial_theta)
Logistic_Regression(X,Y,alpha,initial_theta,iterations)

