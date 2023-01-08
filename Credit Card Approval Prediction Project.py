#!/usr/bin/env python
# coding: utf-8
Credit card applications

Commercial banks receive a lot of applications for credit cards. Many of them get rejected for many reasons, like high loan 
balances, low income levels, or too many inquiries on an individual's credit report, for example. Manually analyzing these 
applications is mundane, error-prone, and time-consuming (and time is money!). Luckily, this task can be automated with the
power of machine learning and pretty much every commercial bank does so nowadays. In this notebook, we will build an automatic
credit card approval predictor using machine learning techniques, just like the real banks do

We'll use the application record data and credit record data . The structure of this notebook is as follows:

First, we will start off by loading and viewing the dataset.
We will see that the dataset has a mixture of both numerical and non-numerical features, that it contains values from different ranges, plus that it contains a number of missing entries.
We will have to preprocess the dataset to ensure the machine learning model we choose can make good predictions.
After our data is in good shape, we will do some exploratory data analysis to build our intuitions.
Finally, we will build a machine learning model that can predict if an individual's application for a credit card will be accepted.
First, loading and viewing the dataset. We find that since this data is confidential, the contributor of the dataset has anonymized the feature names.
# In[93]:


#Importing usefull libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Loading Dataset
apprecord=pd.read_csv("application_record.csv")


# In[3]:


credrecord=pd.read_csv("credit_record.csv")


# In[4]:


#Inspect Data(apprecord)
apprecord.head()


# In[5]:


#Inspect Data(Credrecord)
credrecord.head()


# In[6]:


#Checking the  number of rows and columns in both the tables
apprecord.shape


# In[7]:


credrecord.shape


# As we can see from our first glance at the data, the dataset has a mixture of numerical and non-numerical features.
# This can be fixed with some preprocessing, but before we do that, let's learn about the dataset a bit more to see if
# there are other dataset issues that need to be fixed.

# In[8]:


#As we can see there are  134203 missing the values in OCCUPATION_TYPE column
apprecord.isnull().sum()


# In[9]:


credrecord.isnull().sum()


# In[10]:


apprecord.info()


# In[11]:


credrecord.info()


# In[12]:


# Print summary statistics
apprecord.describe()


# In[13]:


credrecord.describe()


# In[14]:


#Handling the missing values in the column Occupation_Type
apprecord['OCCUPATION_TYPE'].fillna(value='0',inplace=True)


# In[15]:


#As we can see now that there is no missing values in OCCUPATION_TYPE column
apprecord.isnull().sum()


# In[16]:


#Checking Type of data (integer, float, Python object, etc.)
apprecord.dtypes


# In[17]:



apprecord.drop('FLAG_MOBIL',axis=1,inplace=True)


# In[18]:


apprecord.drop_duplicates(subset=['ID']).count()


# In[19]:


apprecord.drop_duplicates(subset=['ID']).count()


# In[20]:


plt.figure(figsize=(10,10))

cols_to_plot = ["CNT_CHILDREN","AMT_INCOME_TOTAL","DAYS_BIRTH","DAYS_EMPLOYED"]
apprecord[cols_to_plot].hist(edgecolor='black', linewidth=1.2)
fig=plt.gcf()
fig.set_size_inches(12,6)


# In[21]:


fig, axes = plt.subplots(1,2)

g1=sns.countplot(y=apprecord.NAME_INCOME_TYPE,linewidth=1.2, ax=axes[0])
g1.set_title("Customer Distribution by Income Type")
g1.set_xlabel("Count")

g2=sns.countplot(y=apprecord.NAME_FAMILY_STATUS,linewidth=1.2, ax=axes[1])
g2.set_title("Customer Distribution by Family Status")
g2.set_xlabel("Count")

fig.set_size_inches(14,5)

plt.tight_layout()


plt.show()


# In[22]:


fig, axes = plt.subplots(1,2)

g1= sns.countplot(y=apprecord.NAME_HOUSING_TYPE,linewidth=1.2, ax=axes[0])
g1.set_title("Customer Distribution by Housing Type")
g1.set_xlabel("Count")
g1.set_ylabel("Housing Type")

g2= sns.countplot(y=apprecord.NAME_EDUCATION_TYPE, ax=axes[1])
g2.set_title("Customer Distribution by Education")
g2.set_xlabel("Count")
g2.set_ylabel("Education Type")

fig.set_size_inches(14,5)

plt.tight_layout()

plt.show()


# In[23]:



def countplot_applicants(features):
    for feature in features:
        plt.figure(figsize=(9, 9))
        ax=sns.countplot(y=apprecord[feature], hue='CODE_GENDER' , data=apprecord)
        plt.legend(loc='best')
        total = len(apprecord[feature])
        for p in ax.patches:
            percentage = '{:.1f}%'.format(100 * p.get_width()/total)
            x = p.get_x() + p.get_width() + 0.02
            y = p.get_y() + p.get_height()/2
            ax.annotate(percentage, (x, y))
    plt.show()


# In[24]:


countplot_applicants(['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE'])


# In[27]:



def countplot_credit(features):
    for feature in features:
        plt.figure(figsize=(9, 9))
        ax=sns.countplot(y=credrecord[feature], data=credrecord)
        total = len(credrecord[feature])
        for p in ax.patches:
            percentage = '{:.1f}%'.format(100 * p.get_width()/total)
            x = p.get_x() + p.get_width() + 0.02
            y = p.get_y() + p.get_height()/2
            ax.annotate(percentage, (x, y))
    plt.show()


# In[28]:


countplot_credit(['STATUS'])


# In[29]:


apprecord.drop(['DAYS_EMPLOYED'], axis=1, inplace=True)


# In[30]:


apprecord.columns


# In[31]:


def Cat_to_Num(features):
    for feature in features:
        feature_list = list(np.unique(apprecord[feature]))
        feature_dict = {}
        for i in range(len(feature_list)):
                       feature_dict[feature_list[i]] = i
        apprecord.replace({feature : feature_dict}, inplace=True)
        print(feature, '-->', feature_dict)
        


# In[32]:


categorical_features = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS', 'OCCUPATION_TYPE', 'NAME_HOUSING_TYPE', 'NAME_EDUCATION_TYPE']
Cat_to_Num(categorical_features)


# In[35]:


apprecord.head(10)


# In[36]:


convert_to = {'C' : 'Good_Debt', 'X' : 'Good_Debt', '0' : 'Good_Debt', '1' : 'Neutral_Debt', '2' : 'Neutral_Debt', '3' : 'Bad_Debt', '4' : 'Bad_Debt', '5' : 'Bad_Debt'}
credrecord.replace({'STATUS' : convert_to}, inplace=True)


# In[37]:


credrecord


# In[ ]:


#Classifing clients with good_debt counts greater than bad_debt as eligible and vice-versa


# In[38]:



credrecord['STATUS2'] = credrecord['STATUS']


# In[39]:


credrecord['STATUS2'].unique()


# In[ ]:



#Replacing C, X, 0 with 'Good_Debt' (C: loan for that month is already paid; X: no loan for that month; 0: loan is 1 to 29 days overdue).¶
#Similarly 1, 2, 3, 4, 5 with 'Bad_Debt' (1: loan is 30 to 59 days overdue; 2: loan is 60 to 89 days overdue; 3: loan is 90 to 119 days overdue;
#4: loan is 120 to 149 days overdue; 5: loan is more than 150 days overdue).


# In[40]:



credrecord = credrecord.replace({'STATUS2' :
                                          {'C' : 'Good_Debt',
                                           'X' : 'Good_Debt',
                                           '0' : 'Good_Debt',
                                           '1' : 'Bad_Debt',
                                           '2' : 'Bad_Debt',
                                           '3' : 'Bad_Debt',
                                           '4' : 'Bad_Debt',
                                           '5' : 'Bad_Debt'}})


# In[41]:


credrecord.value_counts(subset=['ID', 'STATUS2']).unstack(fill_value=0)


# In[42]:



credrecord = credrecord.value_counts(subset=['ID', 'STATUS2']).unstack(fill_value=0).reset_index()


# In[43]:



credrecord


# In[ ]:


#Classifing clients with good_debt counts greater than bad_debt as eligible and vice-versa


# In[45]:


credrecord.loc[(credrecord['Good_Debt'] > credrecord['Bad_Debt']), 'Status'] = 1


# In[46]:


credrecord.loc[(credrecord['Good_Debt'] <= credrecord['Bad_Debt']), 'Status'] = 0


# In[47]:


credrecord['Status'] = credrecord['Status'].astype(int)


# In[48]:



credrecord


# In[49]:



credrecord


# In[ ]:


#Merging Both application_details and credit_record data


# In[50]:


Final_Credit_data = apprecord.merge(credrecord, how='inner', on=['ID'])


# In[51]:


Final_Credit_data.head(10)


# In[52]:


Final_Credit_data.describe()


# In[ ]:



#Splitting the credit_approval_data into training and testing sets


# In[54]:


from sklearn.model_selection import train_test_split


# In[55]:


X = Final_Credit_data.drop('Status', axis=1)


# In[74]:



y =Final_Credit_data['Status']


# In[81]:


#Splitting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X,
                                y,
                                test_size=0.33,
                                random_state=42)


# In[82]:


#scalling the data

# Import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

# Instantiate MinMaxScaler and use it to rescale X_train and X_test
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.transform(X_test)


# In[ ]:


#Fitting a logistic regression model to the train set


# In[83]:


# Import LogisticRegression
from sklearn.linear_model import LogisticRegression

# Instantiate a LogisticRegression classifier with default parameter values
logreg = LogisticRegression()

# Fit logreg to the train set
logreg.fit(rescaledX_train, y_train)


# In[89]:


#Making predictions and evaluating performance


# Import confusion_matrix
from sklearn.metrics import confusion_matrix

# Use logreg to predict instances from the test set and store it
y_pred = logreg.predict(rescaledX_test)

# Get the accuracy score of logreg model and print it
print("Accuracy of logistic regression classifier: ", logreg.score(rescaledX_test, y_test))

# Print the confusion matrix of the logreg model
print(confusion_matrix(y_test, y_pred))


# In[ ]:


#Grid searching and making the model perform better


# In[ ]:


#Our model was pretty good! It was able to yield an accuracy score of almost 99%.

#For the confusion matrix, the first element of the of the first row of the confusion matrix denotes the true negatives meaning the number of negative
#instances (denied applications) predicted by the model correctly. And the last element of the second row of the confusion matrix denotes the true positives 
#meaning the number of positive instances (approved applications) predicted by the model correctly.

#Let's see if we can do better. We can perform a grid search of the model parameters to improve the model's ability to predict credit card approvals.

#scikit-learn's implementation of logistic regression consists of different hyperparameters but we will grid search over the following two:

#tol
#max_iter


# In[91]:


# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Define the grid of values for tol and max_iter
tol = [0.01, 0.001, 0.0001]
max_iter = [100, 150, 200]

# Create a dictionary where tol and max_iter are keys and the lists of their values are corresponding values
param_grid = dict(tol= tol, max_iter= max_iter)


# In[ ]:


#Finding the best performing model
#We have defined the grid of hyperparameter values and converted them into a single dictionary format which GridSearchCV() expects as one of its parameters. Now, we will begin the grid search to see which values perform best.

#We will instantiate GridSearchCV() with our earlier logreg model with all the data we have. Instead of passing train and test sets separately, we will supply X (scaled version) and y. We will also instruct GridSearchCV() to perform a cross-validation of five folds.

#We'll end the notebook by storing the best-achieved score and the respective best parameters.

#While building this credit card predictor, we tackled some of the most widely-known preprocessing steps such as scaling and missing value imputation. We finished with some machine learning to predict if a person's application for a credit card would get approved or not given some information about that person.


# In[92]:


# Instantiate GridSearchCV with the required parameters
grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)

# Use scaler to rescale X and assign it to rescaledX
rescaledX = scaler.fit_transform(X)

# Fit data to grid_model
grid_model_result = grid_model.fit(rescaledX, y)

# Summarize results
best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_
print("Best: %f using %s" % (best_score, best_params))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




