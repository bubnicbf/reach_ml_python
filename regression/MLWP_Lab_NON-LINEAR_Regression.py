#=======================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
% matplotlib inline

#===============================================================================

# Though Linear regression is very good to solve many problems, it cannot be 
# used for all datasets. First recall how linear regression, could model a 
# dataset. It models a linear relation between a dependent variable y and 
# independent variable x. It had a simple equation, of degree 1, for example 
# y = 2*(x) + 3.

x = np.arange(-5.0, 5.0, 0.1)

##You can adjust the slope and intercept to verify the changes in the graph
y = 2*(x) + 3
y_noise = 2 * np.random.normal(size=x.size)
ydata = y + y_noise
#plt.figure(figsize=(8,6))
plt.plot(x, ydata,  'bo')
plt.plot(x, y, 'r')
plt.title('Linear Model')
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()


# ================================================================================
# create a dataframe for export:

# np.savetxt('/Users/jimhomolak/Desktop/random_linear.txt', x, delimiter =', ')
# import pandas as pd
# import numpy as np
# images = np.array(images)
# label = np.array(label)
# dataset = pd.DataFrame({'label': label, 'images': list(images)}, columns=['label', 'images'])

x = x = np.arange(-5.0, 5.0, 0.1)
y_noise = 2 * np.random.normal(size=x.size)
ydata = y + y_noise
y = 2*(x) + 3


# dataset = pd.DataFrame({'x': x, 'y': y, 'ydata':ydata}, columns=['x', 'y','ydata'])
# df.to_csv(sep=' ', index=False, header=False)


# ================================================================================
# Non-linear regressions are a relationship between independent variables
#  𝑥 and a dependent variable 𝑦 which result in a non-linear function modeled
#  data. Essentially any relationship that is not linear can be termed as 
# non-linear, and is usually represented by the polynomial of 𝑘 degrees 
# (maximum power of 𝑥 ).
#  𝑦=𝑎𝑥3+𝑏𝑥2+𝑐𝑥+𝑑 
# Non-linear functions can have elements like exponentials, logarithms, fractions, and others. For example:
# 𝑦=log(𝑥)
# Or even, more complicated such as :
# 𝑦=log(𝑎𝑥3+𝑏𝑥2+𝑐𝑥+𝑑)

# Let's take a look at a cubic function's graph.
x = np.arange(-5.0, 5.0, 0.1)

##You can adjust the slope and intercept to verify the changes in the graph
y = 1*(x**3) + 1*(x**2) + 1*x + 3
y_noise = 20 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata,  'bo')
plt.plot(x, y, 'r')
plt.title('Cubic Model')
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

# Quadratic
# 𝑌=𝑋2

x = np.arange(-5.0, 5.0, 0.1)
##You can adjust the slope and intercept to verify the changes in the graph
y = np.power(x,2)
y_noise = 2 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata,  'bo')
plt.plot(x, y, 'r')
plt.title('Quadratic Model')
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

# EXPONENTIAL
# An exponential function with base c is defined by
# 𝑌=𝑎+𝑏𝑐𝑋
# where b ≠0, c > 0 , c ≠1, and x is any real number. The base, c, 
# is constant and the exponent, x, is a variable.

X = np.arange(-5.0, 5.0, 0.1)

# You can adjust the slope and intercept to verify the changes in the graph
Y= np.exp(X)
plt.plot(X, Y)
plt.title('Exponential Model')
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

# LOGARITHMIC
# The response 𝑦
# is a results of applying logarithmic map from input 𝑥's to output variable 𝑦.
# It is one of the simplest form of log(): i.e. # 𝑦=log(𝑥)

# Please consider that instead of 𝑥# , we can use 𝑋, which can be polynomial
# representation of the 𝑥's. In general form it would be written as 𝑦=log(𝑋)

X = np.arange(-5.0, 5.0, 0.1)
Y = np.log(X)
plt.plot(X, Y)
plt.title('Logarithmic Model')
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()


# Sigmoidal/Logistic
# 𝑌=𝑎+𝑏1+𝑐(𝑋−𝑑)

X = np.arange(-5.0, 5.0, 0.1)
Y = 1-4/(1+np.power(3, X-2))
plt.plot(X, Y)
plt.title('Logistic Model')
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

# ===============================================================================
# Non-Linear Regression example
# For an example, we're going to try and fit a non-linear model to the datapoints 
# corrensponding to China's GDP from 1960 to 2014. We download a dataset with two
#  columns, the first, a year between 1960 and 2014, the second, China's 
# corresponding annual gross domestic income in US dollars for that year.

# we're going to try and fit a non-linear model to the datapoints corrensponding 
# to China's GDP from 1960 to 2014. We download a dataset with two columns, the 
# first, a year between 1960 and 2014, the second, China's corresponding annual 
# gross domestic income in US dollars for that year.

!wget -nv -O china_gdp.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/china_gdp.csv
df = pd.read_csv("china_gdp.csv")
df.head(10)

# Plotting the Dataset
# This is what the datapoints look like. It kind of looks like an either logistic
#  or exponential function. The growth starts off slow, then from 2005 on forward,
#  the growth is very significant. And finally, it deaccelerates slightly in the 2010s.

plt.figure(figsize=(8,5))
x_data, y_data = (df["Year"].values, df["Value"].values)
plt.plot(x_data, y_data, 'ro')
plt.title('China GDP from 1960 to 2014')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

# CHOSING A MODEL
# From an initial look at the plot, we determine that the logistic function could
# be a good approximation, since it has the property of starting with a slow 
# growth, increasing growth in the middle, and then decreasing again at the end; 
# as illustrated below:

X = np.arange(-5.0, 5.0, 0.1)
Y = 1.0 / (1.0 + np.exp(-X))
plt.plot(X,Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

# The formula for the logistic function is the following:
# 𝑌̂ =11+𝑒𝛽1(𝑋−𝛽2)
# 𝛽1 : Controls the curve's steepness,
# 𝛽2 : Slides the curve on the x-axis.


# ===============================================================================
# BUILDING THE MODEL
# Now, let's build our regression model and initialize its parameters.

def sigmoid(x, Beta_1, Beta_2):
     y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
     return y

# Lets look at a sample sigmoid line that might fit with the data:
beta_1 = 0.10
beta_2 = 1990.0

#logistic function
Y_pred = sigmoid(x_data, beta_1 , beta_2)

#plot initial prediction against datapoints
plt.plot(x_data, Y_pred*15000000000000.)
plt.plot(x_data, y_data, 'ro')

# Our task here is to find the best parameters for our model. Lets first 
# normalize our x and y:

# Lets normalize our data
xdata =x_data/max(x_data)
ydata=y_data / max(y_data)

# How we find the best parameters for our fit line?
# we can use curve_fit which uses non-linear least squares to fit our 
# sigmoid function, to data. Optimal values for the parameters so that
# the sum of the squared residuals of sigmoid(xdata, *popt) - ydata is minimized.
# popt are our optimized parameters.

## THIS IS A CURVE FITTING FUNCTION, NOT A REGRESSION MODEL PREDICTION
# CURVE FITTING VS REGRESION PREDICTION


# popt are our optimized parameters
from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid, xdata, ydata)
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))

# OPTIMIZED PARAMETERS: beta_1 = 690.447527, beta_2 = 0.997207

# PLOT THE RESULTING REGRESSION MODEL
# x_data is the normalized Year value
# y_data is the normalized GDF value

# PLOT ACTUAL DATA AGAINST OPTIMIZED LOGISTIC FIT

x = np.linspace(1960, 2015, 55)
x = x/max(x)            ## standardize
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()



# DETERMINE MODEL ACCURACY
# split data into train/test
msk = np.random.rand(len(df)) < 0.8
train_x = xdata[msk]
test_x = xdata[~msk]
train_y = ydata[msk]
test_y=ydata[~msk]


# build the model using train set
popt, pcov=curve_fit(sigmoid, train_x, train_y)

# predict using test set
y_hat=sigmoid(test_x, *popt)

# evaluation
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - test_y)))
# Mean absolute error: 0.27

print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - test_y)** 2))
# Residual sum of squares (MSE): 0.19

from sklearn.metrics import r2_score
print("R2-score: %.2f" % r2_score(y_hat , test_y) )
# R2-score: -141341438754972135192760155455356928.00
