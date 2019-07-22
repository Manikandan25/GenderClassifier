from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier

clf = tree.DecisionTreeClassifier()

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data
#clf = clf.fit(X, Y)

#prediction = clf.predict([[190, 70, 43]])

# CHALLENGE compare their reusults and print the best one!
algorithms=[KNeighborsClassifier,GaussianProcessClassifier,RandomForestClassifier]
algorithmsNames=["KNeighborsClassifier","GaussianProcessClassifier","RandomForestClassifier"]
bestAlgorithm=""
bestAlgorithmName=""
bestScore=0
for i in range(3):
	clf=algorithms[i]()
	clf.fit(X,Y)
	print(algorithmsNames[i],"'s score is",clf.score(X,Y))
	if(clf.score(X,Y)>=bestScore):
		bestAlgorithm=clf
		bestAlgorithmName=algorithmsNames[i]

#print(prediction)
print("The Best Algorithm for this data is :",bestAlgorithmName)
print(bestAlgorithm.predict([[190, 70, 43]]))