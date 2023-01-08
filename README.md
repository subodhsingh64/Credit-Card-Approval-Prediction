
# Credit Card Approval Prediction

A brief description of what this project does and who it's for


## Business Problem
Commercial banks receive a lot of applications for credit cards. Many of them get rejected for many reasons, like high loan 
balances, low income levels, or too many inquiries on an individual's credit report, for example. Manually analyzing these 
applications is mundane, error-prone, and time-consuming (and time is money!). Luckily, this task can be automated with the
power of machine learning and pretty much every commercial bank does so nowadays. In this notebook, we will build an automatic
credit card approval predictor using machine learning techniques, just like the real banks do
## Data Gathering

application_record.csv and credit_record .These are the datasets  for the project,which I have downloaded from Kaggle
Credit Card Approval Prediction Project.ipynb: The jupyter notebook which includes the analysis and modeling
The project also includes a front-end webiste which can be deloyed using flask app and will provide with "Approved" or "Not Approved" status on entering all the parameters
For the front-end refer to the folder Credit Card Approval, first run the model.py and then the app.py file
## Libraries used
sklearn
pandas
numpy
seaborn
matplotlib
## Methods Used
    


- Exploratory Data Analysis:
-   Missing Value Imputation - Droping duplicates and columns
- coversion  categorical data into numeric
- Standardization of Data using MinmaxScaler
- Splitting the credit_approval_data into training and testing sets
- Model Building using Logistic Regression
- Making predictions and evaluating performance of model using  Accuracy Score




## Result
Our model was pretty good! It was able to yield an accuracy score of almost 99%.

For the confusion matrix, the first element of the of the first row of the confusion matrix denotes the true negatives meaning the number of negative
instances (denied applications) predicted by the model correctly. And the last element of the second row of the confusion matrix denotes the true positives 
meaning the number of positive instances (approved applications) predicted by the model correctly.

