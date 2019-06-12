
# coding: utf-8

# In[1]:


"""
CellStrat
"""

#==============================================================================
# Import libraries
#==============================================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#==============================================================================
# import the dataset of flat prices
#==============================================================================

flatdata = pd.read_csv ('Price.csv')
X = flatdata.iloc [:,:-1].values
y = flatdata.iloc [:,1].values

X


# In[2]:

#==============================================================================
# split the dataset into training and test set. We will use 1/3 approach
#==============================================================================

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, 
                                                     random_state = 0)



# In[3]:

#==============================================================================
# Fitting the Linear Regression algo to the Training set
#==============================================================================

from sklearn.linear_model import LinearRegression
regressoragent = LinearRegression()
regressoragent.fit (X_train, y_train )            


# In[5]:

#==============================================================================
# Now check what our model learned by predicting the X_test values
#==============================================================================

predictValues = regressoragent.predict(X_test)

print(y_test - predictValues)

#==============================================================================
# So now let us visualize the Training set
#==============================================================================
plt.scatter(X_train, y_train, color = 'green')
plt.plot (X_train, regressoragent.predict (X_train), color = 'red')
plt.title ('compare Training result - Area/Price')
plt.xlabel('Area of Flat')
plt.ylabel('Price')
plt.show()

#==============================================================================
# So now let us visualize the Test set
#==============================================================================
plt.scatter(X_test, y_test, color = 'green')
plt.plot (X_train, regressoragent.predict (X_train), color = 'red')
plt.title ('compare Test result - Area/Price')
plt.xlabel('Area of Flat')
plt.ylabel('Price')
plt.show()

#==============================================================================
# So now let us visualize the ENTIRE set
#==============================================================================
plt.scatter(X_train, y_train, color = 'green')
plt.scatter(X_test, y_test, color = 'blue')
plt.plot (X_train, regressoragent.predict (X_train), color = 'red')
plt.title ('compare ENTIRE result - Area/Price')
plt.xlabel('Area of Flat')
plt.ylabel('Price')
plt.show()


# In[ ]:




# In[ ]:



