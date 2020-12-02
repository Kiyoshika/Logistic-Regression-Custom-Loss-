"""
Comparing custom logistic regression with sklearn's using make_classification data
"""

import math
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


X, Y = make_classification(n_samples = 500, n_features = 6, n_informative = 2, n_redundant = 0, class_sep = 0.15)

xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.3)

w = [0.1, 0.2, 0.3]

def sigmoid(x):
  return 1 / (1 + 2.71828**(-x))

def eval(X, w):
  res = 0

  for i in range(len(w)):
    if (i == 0):
      res += w[i]
    else:
      res += X[i-1]*w[i]

  return res

def loss(y, yhat):
  return 2.71828**((y-yhat)**3) - 1

def loss_deriv(y, yhat):
  return (loss(y, yhat + 0.001) - loss(y, yhat)) / 0.001

for iter in range(1500):
  for i in range(len(ytrain)):
    change = loss_deriv(ytrain[i], sigmoid(eval(xtrain[i], w)))
    for j in range(len(w)):
      if j == 0:
        w[j] -= 0.5*change
      else:
        w[j] -= 0.5*change*xtrain[i][j-1]

def predict():
  preds = []
  for i in range(len(xtest)):
    if sigmoid(eval(xtest[i], w)) > 0.5:
      preds.append(1)
    else:
      preds.append(0)
  return preds

print(f1_score(ytest, predict()))
print(f1_score(ytest, LogisticRegression(max_iter=2500).fit(xtrain, ytrain).predict(xtest)))
