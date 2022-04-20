"""
Balanced Classes
A variety of class-balancing methods implemented with Random Forests

"""

import sys

# General data processing packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

# Metrics
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# Classifier models
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Image feature packages
import mahotas as mt

plt.rcParams['figure.figsize'] = (7, 7)

"""
Data loading
"""

# Generate Haralick features
def extract_features(image):
    # calculate haralick texture features for 4 types of adjacency
    textures = mt.features.haralick(image)

    # take the mean of it and return it
    ht_mean = textures.mean(axis=0)
    return ht_mean

def get_haralick(input_dataloader):
    haralick_features = []
    y = []
    for batch_features, batch_label in iter(input_dataloader):
        j = 0 
        while j < len(batch_features):
            gray =  np.dot(batch_features[j][...,:3], [76.245, 149.685, 29.07]) #combines unormalisation and 
            haralick_feature = extract_features(gray.astype(int))
            haralick_features.append(haralick_feature)
            y.append(batch_label[j])
            j = j + 1 
    return haralick_features, y

haralick_train, y_train = get_haralick(train_dataloader)
haralick_validate, y_validate = get_haralick(validate_dataloader)


"""
Standard Random Forest without balancing, for comparison
"""
# Standard Random Forest
skBootWeight = RandomForestClassifier(n_estimators=750)
skBootWeight.fit(haralick_train, y_train)
y_pred = skBootWeight.predict(haralick_validate)

print(metrics.f1_score(y_pred,y_validate,average='weighted'))
print(metrics.classification_report(y_pred, y_validate))

# confusion matrix plot
mat = confusion_matrix(y_validate, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')


"""
## Method a) Balanced Random Forest Classifier
Has the best f1 score for class 3, but with a significant cost to overall weighted f1 score.
"""

balForest = BalancedRandomForestClassifier(n_estimators=700)
balForest.fit(haralick_train, y_train)

y_pred = balForest.predict(haralick_validate)

print(metrics.classification_report(y_pred, y_validate))
print(metrics.f1_score(y_pred,y_validate,average='weighted'))

# confusion matrix plot
mat = confusion_matrix(y_validate, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')


"""
## Method b) Sklearn's bootstrap class weighting
Balances class distribution for each tree in the forest.
Does not perform well for improving class 3's predictions.
"""

skBootWeight = RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample')
skBootWeight.fit(haralick_train, y_train)
y_pred = skBootWeight.predict(haralick_validate)

# confusion matrix plot
mat = confusion_matrix(y_validate, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');

# metrics
print(metrics.f1_score(y_pred,y_validate,average='weighted'))
print(metrics.classification_report(y_pred, y_validate))


"""
## Method c) SMOTE oversampling followed by undersampling with Balanced Random Forest Classifier
Seems to work the best in terms of improving the class 3 f1 score without decreasing the overall f1 weighted score too much.
"""

model = BalancedRandomForestClassifier(n_estimators=750, max_features=3)
over = SMOTE()
steps = [('over', over), ('model', model)]
pipeline = Pipeline(steps=steps)
pipeline.fit(haralick_train, y_train)
y_pred = pipeline.predict(haralick_validate)

print(metrics.f1_score(y_pred,y_validate,average='weighted'))
print(metrics.classification_report(y_pred, y_validate))

# confusion matrix plot
mat = confusion_matrix(y_validate, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')


"""Hyperparameter tuning: determining the best number of estimators in the forest"""
estimators = [10,50,100,150,200,500,750,1000]
f1_scores = []
iters = []
n = 0
over = SMOTE()
for estimator in estimators:
    model = BalancedRandomForestClassifier(n_estimators=estimator)
    steps = [('over', over), ('model', model)]
    pipeline = Pipeline(steps=steps)
    scores = 0
    for i in range(0,5):
        pipeline.fit(haralick_train, y_train)
        y_pred = pipeline.predict(haralick_validate)
        scores += metrics.f1_score(y_pred,y_validate,average='weighted')
    
    f1_scores.append(scores/5)
    n = n + 1 
    iters.append(estimator)

# plotting
plt.title("Test Curve")
plt.plot(iters, f1_scores, label="Test")
plt.xlabel("N Estimators")
plt.ylabel("F1 Score")
plt.show()


"""Hyperparameter tuning: determining the best maximum number of features to select in feature subsampling"""
max_feats = range(int(np.sqrt(14)),14)
f1_scores = []
iters = []
n = 0
over = SMOTE()
for nfeat in max_feats:
    model = BalancedRandomForestClassifier(n_estimators=750, max_features=nfeat)
    steps = [('over', over), ('model', model)]
    pipeline = Pipeline(steps=steps)
    scores = 0
    for i in range(0,5):
        pipeline.fit(haralick_train, y_train)
        y_pred = pipeline.predict(haralick_validate)
        scores += metrics.f1_score(y_pred,y_validate,average='weighted')
    
    f1_scores.append(scores/5)
    n = n + 1 
    iters.append(nfeat)

# plotting
plt.title("Test Curve")
plt.plot(iters, f1_scores, label="Test")
plt.xlabel("Max features")
plt.ylabel("F1 Score")
plt.show()


max_feats = range(1,9)
f1_scores = []
iters = []
n = 0
over = SMOTE()
for nfeat in max_feats:
    model = BalancedRandomForestClassifier(n_estimators=750, max_features=nfeat)
    steps = [('over', over), ('model', model)]
    pipeline = Pipeline(steps=steps)
    scores = 0
    for i in range(0,5):
        pipeline.fit(haralick_train, y_train)
        y_pred = pipeline.predict(haralick_validate)
        scores += metrics.f1_score(y_pred,y_validate,average='weighted')
    
    f1_scores.append(scores/5)
    n = n + 1 
    iters.append(nfeat)

# plotting
plt.title("Test Curve")
plt.plot(iters, f1_scores, label="Test")
plt.xlabel("Max features")
plt.ylabel("F1 Score")
plt.show()
