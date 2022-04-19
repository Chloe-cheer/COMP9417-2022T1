import data
from data import get_dataloaders, extract_features, get_features
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier

# extract 64x64 averaging and haralick features
train_dataloader, validate_dataloader = get_dataloaders('y_train.npy', 'X_train.npy')
haralick_train, pool_train, y_train = get_features(train_dataloader)
haralick_validate, pool_validate, y_validate = get_features(validate_dataloader)

# change the datatype of y to be a numpy array, not a tensor
i = 0 
while i < len(y_train):
    y_train[i] = y_train[i].item()
    i = i + 1
i = 0 
while i < len(y_validate):
    y_validate[i] = y_validate[i].item()
    i = i + 1

# check distribution of labels 
see_distribution(y_train)
see_distribution(y_validate)

# check haralick feature distributions
haralick_eda = {}
i = 0 
while i < len(haralick_train):
    if y_train[i] in haralick_eda:
        haralick_eda[y_train[i]] = haralick_eda[y_train[i]] + haralick_train[i]
    else:
        haralick_eda[y_train[i]] = haralick_train[i]
    i = i + 1 
for label in haralick_eda:
    haralick_eda[label] = haralick_eda[label]/len(haralick_train)
    plt.title("Histogram of average feature values for class: " + str(label))
    plt.hist(haralick_eda[label],10)
    plt.show()

# make a normalised copy of haralick features 
scaler = StandardScaler()
scaler.fit(haralick_train)
haralick_train_scaled = scaler.transform(haralick_train)
haralick_validate_scaled = scaler.transform(haralick_validate)

# logistic regression

### haralick features   
lm = linear_model.LogisticRegression(max_iter = 10000)
lm.fit(haralick_train, y_train)
fig, ax = plt.subplots(figsize=(10, 6))
print(metrics.f1_score(lm.predict(haralick_train),y_train,average='weighted'))
print(metrics.f1_score(lm.predict(haralick_validate),y_validate,average='weighted'))
ax.set_title('Haralick Features Log-Regression Confusion Matrx')
disp =metrics.plot_confusion_matrix(lm, haralick_validate, y_validate, ax = ax)
disp.confusion_matrix

### 64x64 averaging 
lm = linear_model.LogisticRegression(max_iter = 10000)
lm.fit(pool_train, y_train)
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title('Neighbour Averaging Features Log-Regression Confusion Matrx')
disp =metrics.plot_confusion_matrix(lm, pool_validate, y_validate, ax = ax)
disp.confusion_matrix
print(metrics.f1_score(lm.predict(pool_train),y_train,average='weighted'))
print(metrics.f1_score(lm.predict(pool_validate),y_validate,average='weighted'))

### normalised haralick features 
lm = linear_model.LogisticRegression(max_iter = 10000)
lm.fit(haralick_train_scaled, y_train)
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title('Normalised Haralick Features Log-Regression Confusion Matrx')
disp =metrics.plot_confusion_matrix(lm, haralick_validate_scaled, y_validate, ax = ax)
disp.confusion_matrix
print(metrics.f1_score(lm.predict(haralick_train_scaled),y_train,average='weighted'))
print(metrics.f1_score(lm.predict(haralick_validate_scaled),y_validate,average='weighted'))

### test different values of Cs...
Cs = [0.0001,0.001,0.01,0.05,0.1,0.5,1,1.5,2]
f1_scores = []
iters = []
n = 0
for C in Cs:
    lm = linear_model.LogisticRegression(max_iter = 10000, C=C)
    lm.fit(haralick_train_scaled, y_train)
    f1_scores.append(metrics.f1_score(lm.predict(haralick_validate_scaled),y_validate,average='weighted'))
    n = n + 1 
    iters.append(C)

plt.title("Test Curve")
plt.plot(iters, f1_scores, label="Test")
plt.xlabel("C")
plt.ylabel("F1 Score")
plt.show()

# gradient boosted classifier

### find the best estimator
estimators = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
f1_scores = []
iters = []
n = 0
for estimator in estimators:
    clf = GradientBoostingClassifier(n_estimators=estimator, learning_rate=1.0, max_depth=1, random_state=0).fit(haralick_train, y_train)
    f1_scores.append(metrics.f1_score(clf.predict(haralick_validate),y_validate,average='weighted'))
    n = n + 1 
    iters.append(estimator)

plt.title("Test Curve")
plt.plot(iters, f1_scores, label="Test")
plt.xlabel("N Estimators")
plt.ylabel("F1 Score")
plt.show()

### find the best learning rate
learning_rates = [0.001,0.01,0.05,0.1,0.5,1,1.5]
f1_scores = []
iters = []
n = 0
for rate in learning_rates:
    clf = GradientBoostingClassifier(n_estimators=80, learning_rate=rate, max_depth=1, random_state=0).fit(haralick_train, y_train)
    f1_scores.append(metrics.f1_score(clf.predict(haralick_validate),y_validate,average='weighted'))
    n = n + 1 
    iters.append(rate)

plt.title("Test Curve")
plt.plot(iters, f1_scores, label="Test")
plt.xlabel("Learning Rates")
plt.ylabel("F1 Score")
plt.show()

### find the best max depth
depths = [1,2,3,4,5,6,7,8,9,10]
f1_scores = []
iters = []
n = 0
for depth in depths:
    clf = GradientBoostingClassifier(n_estimators=80, learning_rate=1, max_depth=depth, random_state=0).fit(haralick_train, y_train)
    f1_scores.append(metrics.f1_score(clf.predict(haralick_validate),y_validate,average='weighted'))
    n = n + 1 
    iters.append(depth)

plt.title("Test Curve")
plt.plot(iters, f1_scores, label="Test")
plt.xlabel("Max Depths")
plt.ylabel("F1 Score")
plt.show() 

### 64x64 averaging 
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title('Neighbour Averaging Features Gradient Boosted Classifier Confusion Matrx')
clf = GradientBoostingClassifier(n_estimators=80, learning_rate=1.0, max_depth=1, random_state=0).fit(pool_train, y_train)
print(metrics.f1_score(clf.predict(pool_train),y_train,average='weighted'))
print(metrics.f1_score(clf.predict(pool_validate),y_validate,average='weighted'))
disp =metrics.plot_confusion_matrix(clf, pool_validate, y_validate, ax = ax)
disp.confusion_matrix

### haralick features
clf = GradientBoostingClassifier(n_estimators=80, learning_rate=1, max_depth=1, random_state=0).fit(haralick_train, y_train)
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title('Haralick Features Gradient Boosted Classifier Confusion Matrx')
disp =metrics.plot_confusion_matrix(clf, haralick_validate, y_validate, ax = ax)
disp.confusion_matrix
print(metrics.f1_score(clf.predict(haralick_train),y_train,average='weighted'))
print(metrics.f1_score(clf.predict(haralick_validate),y_validate,average='weighted'))