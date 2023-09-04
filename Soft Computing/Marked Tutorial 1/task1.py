from sklearn.neural_network import MLPClassifier
from pandas import read_csv as rc
from pandas import DataFrame as df
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
#from seaborn import kdeplot
from sklearn.metrics import confusion_matrix
from seaborn import set, heatmap


dataset_init = rc('Marked Tutorial 1\data\cancer.csv', names=['mean_radius','mean_texture','mean_perimeter','mean_area','mean_smoothness', 'diagnosis'])

learning_rate = 0.0001
epoch_val = 750


dataset=dataset_init.values
X=dataset[:,0:5] #Attributes for each data entry
Y=dataset[:,5] #True outputs - the correct output ( Correct Iris Species )

mlp=MLPClassifier(hidden_layer_sizes=(100,), 
                  activation='relu',
                  solver='adam',
                  alpha=learning_rate,
                  max_iter=epoch_val
                  )
mlp.fit(X,Y)
pred=mlp.predict(X)
score=mlp.score(X,Y)


cm=confusion_matrix(Y, 
                    pred, 
                    labels=[1, 0])
df_cm = df(cm, range(2), range(2))
set(font_scale=1.4)
heatmap(df_cm, annot=True, annot_kws={"size": 20})
pyplot.xlabel("Predicted Output")
pyplot.ylabel("True Output")
pyplot.show()

x_train, x_val, y_train, y_val=train_test_split(X, 
                                                Y, 
                                                test_size=0.3,
                                                random_state=0)

#Training set
mlp=MLPClassifier(hidden_layer_sizes=(100,), 
                  activation='relu',
                  solver='adam', 
                  alpha=learning_rate, 
                  max_iter=epoch_val)
mlp.fit(x_train,y_train) #returns a trained mlp model in form of object named 'mlp'
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

