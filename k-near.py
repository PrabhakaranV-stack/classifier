"""To build a classifier and to predict the unknown clases
		in the telecommunication data by using the classification 
				algorithm K-NEAREST NEIGHBOUR

											""" 
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing,metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
%matplotlib inline

#We use !wget to download the data
df = pd.read_csv('teleCust1000t.csv')
df.head(6)

df['custcat'].value_counts()

df.hist(column='income', bins=50)

#Feature Selection
df.columns

X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
X[0:5]

y = df['custcat'].values
y[0:5]

#Normalize the Data
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

#Training
k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh

#predicting
yhat = neigh.predict(X_test)
yhat[0:5]

#Accuracy Evaluation
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

#using the same model again with k=6
"""k=6
neigh1=KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)
yhat1=neigh1.predict(X_test)
print("Train set Accuracy: ",metrics.accuracy_score(y_train,neigh1.predict(X_train)))
print("Test set Accuracy: ",metrics.accuracy_score(y_test,yhat1))
 
"""

#ACCURACY OF  KNN  WITH DIFFERENT KS

Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc

#PLOT MODEL FOR ACCURACY FOR DIFFERENT NUMBER OF NEIGHBOURS
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)
 