import sklearn
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("TkAgg")


from matplotlib import pyplot as plt
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

##data variable will be a numpy array of shape (150,4) having 150 samples each having four different attributes:sepal-length, sepal-width, petal-length and petal-width 
data = load_iris().data
#print(data)

##there should be 3 classes: Iris-Setosa, Iris-Versicolor, and Iris-Virginica. with 50 samples in each class
##.target is where the classes data are stored has shape of: (150,)
irisSpecies = load_iris().target
#print(irisSpecies)

##data is (150,1) array, this step converts irisSpecies to (150,1) array so we can perform computations between data and irisSpecies
irisSpecies = np.reshape(irisSpecies,(150,1))
#print('reshaped: \n',irisSpecies)

##merge two arrays
data = np.concatenate([data,irisSpecies],axis=-1)



##creating more organized dataframe:
columnName = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'species']
df = pd.DataFrame(data,columns=columnName)

df['species'].replace(0, 'Iris-setosa',inplace=True)
df['species'].replace(1, 'Iris-versicolor',inplace=True)
df['species'].replace(2, 'Iris-virginica',inplace=True)
#print(df.head(5))


#scatter plot of petal width versus length
plt.figure(4, figsize=(8, 8))
plt.scatter(data[:50, 2], data[:50, 3], c='r', label='Iris-setosa')
plt.scatter(data[50:100, 2], data[50:100, 3], c='g',label='Iris-versicolor')
plt.scatter(data[100:, 2], data[100:, 3], c='b',label='Iris-virginica')
plt.xlabel('Petal length',fontsize=15)
plt.ylabel('Petal width',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Petal length vs. Petal width',fontsize=15)
plt.legend(prop={'size': 20})
plt.show()


##splitting data into 80% for training and 20% for testing the algorithm
##random_state of int 32 means the samplings are random but each time you run this, the samples you select for test will stay the same.
train_data,test_data,train_label,test_label = train_test_split(df.iloc[:,2:4], df.iloc[:,4], test_size=0.2, random_state=32) 
#print (df.iloc[:,2:4])
#print(df.iloc[:,4])

##testing optum value of k
# neighbors = np.arange(1,9)
# train_accuracy =np.zeros(len(neighbors))
# test_accuracy = np.zeros(len(neighbors))

# for i,k in enumerate(neighbors):
#     knn = KNeighborsClassifier(n_neighbors=k)

#     #Fit the model
#     knn.fit(train_data, train_label)

#     #Compute accuracy on the training set
#     train_accuracy[i] = knn.score(train_data, train_label)

#     #Compute accuracy on the test set
#     test_accuracy[i] = knn.score(test_data, test_label)

# plt.figure(figsize=(10,6))
# plt.title('KNN accuracy with varying number of neighbors',fontsize=20)
# plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
# plt.plot(neighbors, train_accuracy, label='Training accuracy')
# plt.legend(prop={'size': 20})
# plt.xlabel('Number of neighbors',fontsize=20)
# plt.ylabel('Accuracy',fontsize=20)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.show()



##the final kNN algorithm
knn = KNeighborsClassifier(n_neighbors=3)

#Fit the model
knn.fit(train_data, train_label)

#Compute accuracy on the training set
train_accuracy = knn.score(train_data, train_label)

#Compute accuracy on the test set
test_accuracy = knn.score(test_data, test_label)


#testing variables
predicted= knn.predict([[1,2]]) 
print(predicted)
