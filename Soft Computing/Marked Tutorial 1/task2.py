from sklearn.neural_network import MLPClassifier
from pandas import read_csv as rc
from pandas import DataFrame as df
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
#from seaborn import kdeplot
from sklearn.metrics import confusion_matrix
from seaborn import set, heatmap


dataset_init = rc('Marked Tutorial 1\data\card.csv', names=['time',
                                                             'v1','v2','v3','v4','v5','v6','v7','v8','v9',
                                                             'v10','v11','v12','v13','v14','v15','v16','v17','v18','v19',
                                                             'v20','v21','v22','v23','v24','v25','v26','v27','v28',
                                                             'class'])

dataset=dataset_init.values
X=dataset[:,0:29] #Attributes for each data entry
Y=dataset[:,29] #True outputs

mlp=MLPClassifier(hidden_layer_sizes=(100,), 
                  activation='relu',
                  solver='adam',
                  alpha=0.0001,
                  max_iter=800
                  )
mlp.fit(X,Y)
pred=mlp.predict(X)
score=mlp.score(X,Y)


cm=confusion_matrix(Y, 
                    pred, 
                    labels=[1,0])
df_cm = df(cm, range(2), range(2))
set(font_scale=1.4)
heatmap(df_cm, annot=True, annot_kws={"size": 20})
pyplot.xlabel("Predicted Output")
pyplot.ylabel("True Output")
pyplot.show()


train_ratio = 0.33
validation_ratio = 0.33
test_ratio = 0.33

# train is now 33% of the entire data set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1 - train_ratio)

# test is now 33% of the initial data set
# validation is now 33% of the initial data set
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 

print(x_train, x_val, x_test)


# x_train, x_val, y_train, y_val=train_test_split(X, 
#                                                 Y, 
#                                                 test_size=0.3,
#                                                 random_state=0)

#Training set
mlp=MLPClassifier(hidden_layer_sizes=(100,), 
                  activation='relu',
                  solver='adam', 
                  alpha=0.0001, 
                  max_iter=800)
mlp.fit(x_train,y_train)
predictions_train=mlp.predict(x_train)
accuracy_train=mlp.score(x_train, y_train)
print ("Training set accuracy : ", accuracy_train)


#Validation set
predictions_val=mlp.predict(x_val)
accuracy_val=mlp.score(x_val, y_val)
print ("Validation set accuracy : ",accuracy_val)


# ## Visualising Data ##

# #Data Visualization
# #Histograms
# dataset_init.hist()
# pyplot.show()


# #Scatter_matrix
# scatter_matrix(dataset_init)
# pyplot.show()