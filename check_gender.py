from sklearn import tree

# [height, weight, shoe size]

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37],
		[166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40],
		[159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'female', 'female', 'female', 'male', 'male',
		'male', 'female', 'male', 'female', 'male']

# Define a variable to store the decision tree of our model.
clf = tree.DecisionTreeClassifier()

# training our tree on the given dataset. `fit` method trains our dataset.
clf = clf.fit(X, Y)

# predict using `predict` method.
prediction = clf.predict([[190, 70, 43]])

print(prediction)