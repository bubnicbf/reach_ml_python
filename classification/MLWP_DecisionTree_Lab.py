## ===============================================================================
import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# The Classification and Regression (C&R) Tree node generates a decision tree 
# that allows you to predict or classify future observations. The method uses 
# recursive partitioning to split the training records into segments by minimizing 
# the impurity at each step, where a node in the tree is considered “pure” if 100% 
# of cases in the node fall into a specific category of the target field. 
# Target and input fields can be numeric ranges or categorical (nominal, ordinal, 
# or flags); all splits are binary (only two subgroups). 




!wget -O drug200.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/drug200.csv

my_data = pd.read_csv("drug200.csv", delimiter=",")
my_data[0:5]

my_data.shape
my_data.dtypes
# Age              int64
# Sex             object
# BP              object
# Cholesterol     object
# Na_to_K        float64
# Drug            object


my_data['Age'].values
type(my_data)

# write my_data  dataframe to desktop
my_data.to_csv('/Users/jimhomolak/Desktop/drug200.txt',sep=',', index=False)


# -----------------------------------------------------
# Using my_data as the Drug.csv data read by pandas, declare the following variables:
#     X as the Feature Matrix (data of my_data)
#     <li> <b> y </b> as the <b> response vector (target) </b> </li>
# Remove the column containing the target name since it doesn't contain numeric values.

X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
X[0:5]

type(X)
# numpy.ndarray


# -----------------------------------------------------
# As you may figure out, some featurs in this dataset are catergorical such as Sex or BP. 
# Unfortunately, Sklearn Decision Trees do not handle categorical variables. But still 
# we can convert these features to numerical values. pandas.get_dummies() 
# Convert categorical variable into dummy/indicator variables.

# INDEX FOR FIRST 10 ROWS OF AGE, SEX, BP
# X[rows,cols], 0 based indexing
X[0:10, 1:4]

# -----------------------------------------------------
# array "categorical" independent variables, column dinstict value counts
X[:,1]
np.array(np.unique(X[:,1], return_counts=True)).T

# array column BP dinstict value count
X[:,2]
np.array(np.unique(X[:,2], return_counts=True)).T

# array column Cholesterol dinstict value count
X[:,3]
np.array(np.unique( X[:,3],return_counts=True)).T

# -----------------------------------------------------
# convert categorical values to numeric "dummy values"
from sklearn import preprocessing

le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 
np.array(np.unique(X[:,1],return_counts=True)).T

le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])
np.array(np.unique(X[:,2],return_counts=True)).T

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 
np.array(np.unique(X[:,3],return_counts=True)).T

# verify value conversion
X[0:5]

# -----------------------------------------------------
# Now we can fill the target variable.
y = my_data["Drug"]
np.array(np.unique(my_data["Drug"],return_counts=True)).T


# -----------------------------------------------------
# Setting up the Decision Tree

# -----------------------------------------------------
# We will be using train/test split on our decision tree. 
# Let's import train_test_split from sklearn.cross_validation.

from sklearn.model_selection import train_test_split

# Now train_test_split will return 4 different parameters. We will name them:
# X_trainset, X_testset, y_trainset, y_testset
# The train_test_split will need the parameters:
# X, y, test_size=0.3, and random_state=3.
# The X and y are the arrays required before the split, 
# the test_size represents the ratio of the testing dataset, 
# and the random_state ensures that we obtain the same splits.

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

X_testset.shape      # (60,5)
y_testset.shape      # (60,)
X_trainset.shape     # (140, 5)
y_trainset.shape  # (140,)



# -----------------------------------------------------
# Modeling
# We will first create an instance of the DecisionTreeClassifier called drugTree.
# Inside of the classifier, specify criterion="entropy" so we can see the information gain of each node.

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # it shows the default parameters


# drugTree...
# DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=4,
#             max_features=None, max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf='deprecated', min_samples_split=2,
#             min_weight_fraction_leaf='deprecated', presort=False,
#             random_state=None, splitter='best')


# Next, we will fit the data with the training feature matrix X_trainset 
# and training response vector y_trainset

drugTree.fit(X_trainset, y_trainset)

# Prediction
# Let's make some predictions on the testing dataset and store it into a variable called predTree.

predTree = drugTree.predict(X_testset)

print (predTree [0:5])
# 40     drugY
# 51     drugX
# 139    drugX
# 197    drugX
# 170    drugX
# Name: Drug, dtype: object

print (y_testset [0:5])
# ['drugY' 'drugX' 'drugX' 'drugX' 'drugX']

# Evaluation
# Next, let's import metrics from sklearn and check the accuracy of our model.

from sklearn import metrics
import matplotlib.pyplot as plt

print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))
# DecisionTrees's Accuracy:  0.9833333333333333

# Accuracy classification score computes subset accuracy: the set of labels
#  predicted for a sample must exactly match the corresponding set of labels in y_true.
# In multilabel classification, the function returns the subset accuracy. 
# If the entire set of predicted labels for a sample strictly match with 
# the true set of labels, # then the subset accuracy is 1.0; otherwise it is 0.0.


# Accuracy: The number of correct predictions made divided by the total number of predictions made.

predTree.shape
np.array(np.unique(predTree[:], return_counts=True)).T

# array([['drugA', 7],
#        ['drugB', 5],
#        ['drugC', 5],
#        ['drugX', 20],
#        ['drugY', 23]], dtype=object)

np.array(np.unique(y_testset[:], return_counts=True)).T

# array([['drugA', 7],
#        ['drugB', 5],
#        ['drugC', 5],
#        ['drugX', 21],
#        ['drugY', 22]], dtype=object)


# SO, y_testset = ACTUAL Values & predTest = predictied values
# therefore ther were 

# correct values count = 7 + 5 + 5 + 20 + 22  = 59
# accuracy = 59 / 60 = .983333

# note correct value cannot be greater than the total of the test values
# predicted values = count of correct prediction



## Entropy : 
# A decision tree is built top-down from a root node and involves 
#partitioning the data into subsets that contain instances with similar values
#(homogeneous). ID3 algorithm uses entropy to calculate the homogeneity of 
# a sample. If the sample is completely homogeneous the entropy is zero and
#if the sample is equally divided then it has entropy of one.

# A branch with entropy more than 0 needs further splitting.
# A branch with entropy of 0 is a leaf node (all vallues in the node are the same).


# Visualization
# Lets visualize the tree

from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
%matplotlib inline 

dot_data = StringIO()
filename = "drugtree.png"
featureNames = my_data.columns[0:5]
targetNames = my_data["Drug"].unique().tolist()
out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')