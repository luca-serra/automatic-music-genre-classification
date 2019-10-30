import numpy as np
from sklearn import model_selection
import pandas as pd
import csv as csv
import re
from sklearn import preprocessing, svm, tree, ensemble, decomposition
from random import random
import sklearn.svm as skSVM
from sklearn.datasets import make_classification
import librosa
import pywt
import pywt.data
from numpy.fft import fft, fftshift

def importCSV(nomfichier):
    table = [[0] for i in range(3000)]
    i = -1
    with open(nomfichier, "r") as fichier:
        for row in fichier:
            if row.split(",")[0] == 'Tempo':
                a = 0
            else:
                a=1
                i += 1
            ligne = []
            for j in range(15):
                if a==1:
                    ligne += [float(row.split(",")[j])]
                table[i] = ligne
    return (table)
featurenames = ['Tempo', 'MFCC average', 'MFCC variance',
                                 'Spectral centroid average', 'Spectral centroid variance',
                                 'Spectral bandwidth average', 'Spectral bandwidth variance',
                                 'Spectral roll-off average', 'Spectral roll-off variance', 'Spectral contrast average',
                                 'Spectral contrast variance', 'Spectral flatness average',
                                 'Spectral flatness variance', 'Zero crossing rate average',
                                 'Zero crossing rate variance']
data = importCSV("extracted_features.csv")
df = pd.DataFrame(data, columns=featurenames)

df = df.drop(columns=[])
X0 = preprocessing.scale(df) # Normalize the numerical data so that it can be processed by classifiers
y = [int(i/300) for i in range(len(X0))] # attributing the labels to the dataset (each original audio file has been divided in 3 3.5s samples, which is why we have 3000 data points)


def svd_decomp(data_matrix, dims_to_remove): # SVD decomposition to reduce dimensionality of the data
    U, S, V = np.linalg.svd(data_matrix)
    S = np.diag(S)
    k = len(S)
    r = k - dims_to_remove
    U = U[:, 0:r]
    S = S[0:r, 0:r]
    V = np.transpose(V)[0:r]
    reduced_matrix = np.dot(U, np.dot(S, V))
    return reduced_matrix
def trace_carre(M):
    n=len(M)
    trace=0
    for i in range(n):
        trace+=M[i,i]**2
    return trace

# automatically finding the appropriate number of dimensions to remove, in order to keep ~90% of the energy of the features (rule of thumb)
sigma = np.diag(np.linalg.svd(X0)[1])
n=len(sigma)
dim_remove = 0
energy = trace_carre(sigma) 
energy_reduced = trace_carre(sigma[0:n-1, 0:n-1])
while energy_reduced/energy > 0.9 :
        dim_remove +=1
        energy_reduced = trace_carre(sigma[0:n-dim_remove, 0:n-dim_remove])
        if energy_reduced/energy < 0.85 :
            dim_remove -=1
X = svd_decomp(X0, 0)

print("Dimensions to remove: "+ str(dim_remove))

# randomizing the dataset so that the samples are not grouped by genre anymore
indice = np.random.permutation([i for i in range(len(X))])
X_randomized = np.array([X[indice[i]] for i in range(len(X))])
y_randomized = np.array([y[indice[i]] for i in range(len(X))])

# Initialize cross validation with the desired k value
k = 10
kf = model_selection.KFold(n_splits=k)
cross_val_sets = kf.split(X_randomized)

totalInstances = 0 # Variable that will store the total instances that will be tested  
totalCorrect = 0 # Variable that will store the correctly predicted instances

# With a Linear Kernel SVM
# clf = skSVM.LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#        intercept_scaling=1, loss='squared_hinge', max_iter=1000,
#        multi_class='ovr', penalty='l2', random_state=0, tol=1e-05, verbose=0)

# With a Gaussian Kernel SVM
clf = skSVM.SVC(C=10, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None,
    verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)

cross_validation_accuracies = []
confusion_matrix = np.zeros((10, 10)).astype(int)

for trainIndex, testIndex in cross_val_sets:
    train_Index=[]
    test_Index=[]
    for i in range(trainIndex.size):
        train_Index.append(int(trainIndex[i]))
    for i in range(testIndex.size):
        test_Index.append(int(testIndex[i]))

    trainSet = X_randomized[trainIndex]
    testSet = X_randomized[testIndex]

    trainLabels = y_randomized[trainIndex]
    testLabels = y_randomized[testIndex]
    
    clf.fit(trainSet, trainLabels)
    
    predictedLabels = clf.predict(testSet)

    # displaying the statistics
    correct = 0
    for i in range(len(testSet)):
        if predictedLabels[i] == testLabels[i]:
            correct += 1
        confusion_matrix[testLabels[i]][predictedLabels[i]] += 1
    
    accuracy = float(correct)/(len(testLabels))
    print('Accuracy: ' + str(accuracy))
    cross_validation_accuracies.append(accuracy)

# Accuracy over the whole cross-validation
total_accuracy = 0
for i in cross_validation_accuracies :
    total_accuracy += i
total_accuracy /= k

confusion_matrix = pd.DataFrame(confusion_matrix, columns=['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz','metal','pop','reggae', 'rock'], index=['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz','metal','pop','reggae', 'rock'])

print('\nTotal Accuracy of SVM classiier in k-fold cross-validation (k = ' + str(k) + ') : ' + str(total_accuracy) + '\n')

print('Confusion matrix over the whole cross-validation (lines are the actual label, columns are the predicted label) :')
print(confusion_matrix.to_string())