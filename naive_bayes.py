import numpy as np
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Features (Encoded via probability)
# flowerFV = ['Flowers', 'Beautiful', 'Red', 'White', 'Pink', 'Yellow', 'Plant', 'Blooms', 'Roses', 'Flower']
# fruitsFV = ['Fruits', 'Carbohydrate', 'Fruit', 'Sugar', 'Diabetes', 'Carbs', 'Vitamin', 'Grams', 'Carbs']
# stationaryFV = ['Pen', 'Pens', 'Pencil', 'Pencils', 'Mechanical', 'Ballpoint', 'Writing', 'Lead', 'Fountain', 'Quality']

flFV = [354/1213, 117/1213, 37/1214, 34/1213, 26/1213, 13/1213, 271/1213, 230/1213, 11/1213, 120/1213]
frFV = [88/911, 62/911, 399/911, 97/911, 36/911, 50/911, 143/911, 36/911]
stFV = [585/2046, 345/2046, 203/2046, 155/2046, 150/2046, 117/2046, 196/2046, 159/2046, 97/2046, 39/2046]

# Labels
labels = ['Flowers', 'Fruits', 'Stationary']
labelsEncoded = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]

# Encoding Features & Labels
# le = preprocessing.LabelEncoder()
# flowerEncoded = le.fit_transform(flowerFV)
# fruitsEncoded = le.fit_transform(fruitsFV)
# stationaryEncoded = le.fit_transform(stationaryFV)
# label = le.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(stFV, labelsEncoded, test_size=0.3, random_state=109)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

X_train = X_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

#features = zip(flowerEncoded, fruitsEncoded, stationaryEncoded)
# print(set(features))

# Create a Gaussian Model
gnb = GaussianNB()

# Train the model using the training sets
gnb.fit(X_train, y_train.ravel())

# Predict the response for test dataset
y_pred = gnb.predict(X_test)

print("Gaussian Accuracy:", metrics.accuracy_score(y_test, y_pred))
