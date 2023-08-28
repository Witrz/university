from sklearn.datasets import load_iris
iris=load_iris()

features=iris.data
labels=iris.target



print("Figure 1: show the relationship between Sepal Length and Width")
import matplotlib.pyplot as plt
labels_names = ['I.setosa', 'I.versicolor', 'I.virginica']
colors=['blue', 'red', 'green']
for i in range(len(colors)):
    px=features[:,0][labels==i]
    py=features[:,1][labels==i]
    plt.scatter(px, py, c=colors[i])
plt.legend(labels_names)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()


print("Figure 2: show the relationship between Petal Length and Width")
for i in range(len(colors)):
    px=features[:,2][labels==i]
    py=features[:,3][labels==i]
    plt.scatter(px, py, c=colors[i])
plt.legend(labels_names)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()

from sklearn.decomposition import PCA

print("Figure 3: viewing the Principle Components")
est=PCA(n_components=2)
x_pca=est.fit_transform(features)
colors=['black', 'orange', 'pink']
for i in range(len(colors)):
    px=x_pca[:,0][labels==i]
    py=x_pca[:,1][labels==i]
    plt.scatter(px, py, c=colors[i])
plt.legend(labels_names)
plt.xlabel('First Principle Component')
plt.ylabel('Second Principle Component')
plt.show()

print("Splitting the Test Data into, x_train, x_test, y_train, y_test")
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_pca, labels, test_size=0.4, random_state=33)

from sklearn.svm import SVC
print("Choosing a classifier to train our data using Support Vector Machine")
clf=SVC()
clf.fit(x_train, y_train)
SVC(C=1.0, 
    cache_size=200, 
    class_weight=None, 
    coef0=0.0,
    decision_function_shape=None, 
    degree=3, 
    gamma='auto', 
    kernel='rbf',
    max_iter=-1, 
    probability=False, 
    random_state=None, 
    shrinking=True,
    tol=0.001, 
    verbose=False)

pred = clf.predict(x_test)
#Evaluate the results
from sklearn import metrics
print(metrics.classification_report(y_test, pred, target_names=labels_names))