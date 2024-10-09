#EDA
import numpy as np
import pandas as pd 

#data visualations
import matplotlib.pyplot as plt
import seaborn as sns

#Getting  data
telco_base_data=pd.read_csv(r'D:\Data_Science&AI\ClassRoomMaterial\October\4th_Customer_churn_project\Telco-Customer-Churn.csv')
telco_base_data.head()

#shape of data
telco_base_data.shape

#data information
telco_base_data.info()

#find the unique values for all the attributes
for col in telco_base_data.columns:
    print("column:{} - Unique values: {}".format(col,telco_base_data[col].unique()))
  
#Get column names    
telco_base_data.columns.values

print(telco_base_data.TotalCharges)

#In the dataset the TotalCharges is in Object type but the values are in numeric so 
#convert TotalCharges to numeric
telco_base_data.TotalCharges = pd.to_numeric(telco_base_data.TotalCharges, errors='coerce') #error = coerce (covert to float)

#after converting Object to float
print("TotalCharges value data type:",telco_base_data.TotalCharges.dtype)

#descriptive data
telco_base_data.describe()

print("DropOut Count:", telco_base_data['Churn'].value_counts())

#Plot or Visualize the total DropOut or Churn
telco_base_data['Churn'].value_counts().plot(kind='barh')
plt.xlabel("Count")
plt.ylabel("Churn")
plt.title("Count of Churn(DropOut)")
plt.gca().invert_yaxis() 
plt.show()

#Find in Percentage
telco_base_data['Churn'].value_counts()/len(telco_base_data)

#Gives column infomation
telco_base_data.info(verbose=True)

#Copy the data 
telco_data=telco_base_data.copy()

#check in which TotalCharges column to find null values
telco_data.loc[telco_data['TotalCharges'].isna()==True] #11 data have null values(NaN)

#percentage of null values
telco_data.isna().sum()/len(telco_data)

#Missing Values treatment
#Removing missing values
telco_data.dropna(how = 'any', inplace = True)

#percentage of null values
telco_data.isna().sum()/len(telco_data)

#Divide customers into bins based on tenure e.g. for tenure < 12 months: assign a
# tenure group if 1-12, for tenure between 1 to 2 Yrs, tenure group of 13-24

telco_data['tenure'].value_counts()

#Get the max tenure
print(telco_data['tenure'].max())

# Define the bins and labels
bins = [0, 12, 24, 36, 48, 60, 72] #months
labels = ['1 - 12', '13 - 24', '25 - 36', '37 - 48', '49 - 60', '61 - 72'] #dividing
# Create the tenure_group column
telco_data['tenure_group'] = pd.cut(telco_data['tenure'],bins=bins,labels=labels,right=False)

#Count the values of tenure group
telco_data['tenure_group'].value_counts()

#Percentage of tenure_group
telco_data['tenure_group'].value_counts()/len(telco_data)

#drop column customerID and tenure(unwanted columns)
telco_data.drop(columns= ['customerID','tenure'], axis=1, inplace=True)
telco_data.head()

#Visualize univariate Analysis
for i,predictor in enumerate(telco_data.drop(columns=['Churn','TotalCharges','MonthlyCharges'])):
    plt.figure(i)
    plt.title(f'Churn vs {predictor}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    sns.countplot(data = telco_data,x=predictor,hue='Churn',width=.6)

#2. Convert the target variable 'Churn' in a binary numeric variable i.e. Yes=1 ; No = 0
telco_data['Churn'] = np.where(telco_data.Churn == 'Yes',1,0)

print(telco_data.Churn)

telco_data.dtypes #Churn converted Object to int32

#3. Convert all the categorical variables into dummy variables

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le
print(telco_data.columns)



categ=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
       'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
       'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges',
       'TotalCharges', 'Churn', 'tenure_group']

#Convert all categorical to numeric values
#telco_data[categ]= telco_data[categ].apply(le.fit_tranform)

for col in categ:
    telco_data[col]=le.fit_transform(telco_data[col])
    
#Visualiza the data of TotalCharge vs Monthly charges
sns.boxplot(data=telco_data[['TotalCharges','MonthlyCharges']])

#*9. * Relationship between Monthly Charges and Total Charges
sns.lmplot(data=telco_data, x='MonthlyCharges', y='TotalCharges', fit_reg=False)

#Churn by Monthly Charges and Total Charges
# kernel density estimate (KDE) plot.
# kernel density estimate (KDE) plot.
Mth = sns.kdeplot(telco_data.MonthlyCharges[(telco_data["Churn"] == 0) ],
                color="Red", fill = True)
Mth = sns.kdeplot(telco_data.MonthlyCharges[(telco_data["Churn"] == 1) ],
                ax =Mth, color="Blue", fill= True)
Mth.legend(["No Churn","Churn"],loc='upper right')
Mth.set_ylabel('Density')
Mth.set_xlabel('Monthly Charges')
Mth.set_title('Monthly charges by churn')

#Churn is high when Monthly Charges are high
Tot = sns.kdeplot(telco_data.TotalCharges[(telco_data["Churn"] == 0) ],
                color="Red", fill = True,)
Tot = sns.kdeplot(telco_data.TotalCharges[(telco_data["Churn"] == 1) ],
                ax =Tot, color="Blue", fill= True)
Tot.legend(["No Churn","Churn"],loc='upper right')
Tot.set_ylabel('Density')
Tot.set_xlabel('Total Charges')

corr = telco_data.corr()
print(corr)

# Build a corelation of all predictors with 'Churn' 
plt.figure(figsize=(20,8))
telco_data.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')

#HeatMap
plt.figure(figsize=(15,15))
sns.heatmap(corr,cmap='Set1')

#Bivariate Analysis
new_df1_target0=telco_data.loc[telco_data["Churn"]==0]
new_df1_target1=telco_data.loc[telco_data["Churn"]==1]
def uniplot(df,col,title,hue =None):
    
    sns.set_style('whitegrid')
    sns.set_context('talk')
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['axes.titlepad'] = 30
    
    
    temp = pd.Series(data = hue)
    fig, ax = plt.subplots()
    width = len(df[col].unique()) + 7 + 4*len(temp.unique())
    fig.set_size_inches(width , 8)
    plt.xticks(rotation=45)
    plt.yscale('log')
    plt.title(title)
    ax = sns.countplot(data = df, x= col, order=df[col].value_counts().index,hue = hue,palette='bright') 
        
    plt.show()
uniplot(new_df1_target1,col='Partner',title='Distribution of Gender for DropOut Customers',hue='gender')
uniplot(new_df1_target0,col='Partner',title='Distribution of Gender for Non DropOut Customers',hue='gender')
uniplot(new_df1_target1,col='PaymentMethod',title='Distribution of Payment Method for Dropout Customers',hue='gender')
uniplot(new_df1_target1,col='Contract',title='Distribution of Contract for Dropout Customers',hue='gender')
uniplot(new_df1_target1,col='TechSupport',title='Distribution of TechSupport for Dropout Customers',hue='gender')
uniplot(new_df1_target1,col='SeniorCitizen',title='Distribution of SeniorCitizen for Dropout Customers',hue='gender')

#-------------------------------------------------------------------------------
#Model Training
X= telco_data.drop('Churn',axis =1) # independent variable,X_train , X_test
y = telco_data['Churn']

#data is highly imbalancing
telco_data['Churn'].value_counts()/len(telco_data)

#-------------------------------------------------------
#Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size= 0.2,random_state=42)

print('Traing data shape')
print(X_train.shape)
print(y_train.shape)
print('Testing Data shape')
print(X_test.shape)

print(y_test.value_counts())
print(y_train.value_counts())

#--------------------------------------------------------------------------
#Using Classification
from sklearn.tree import DecisionTreeClassifier
model_dtc = DecisionTreeClassifier(criterion = 'gini',random_state=100, max_depth = 6,min_samples_leaf=8)
model_dtc.fit(X_train,y_train)

#Model Accuracy
model_dtc.score(X_test,y_test)

#Prediction 
y_pred=model_dtc.predict(X_test)

#prdict olny 1 -10 churn
print(y_pred[:10])
#-----------------------------------------------------------------------------
#Classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, labels=[0,1]))

#-------------------------------------------------------
#Visualize Pie Chart
#OverSampling
from imblearn.over_sampling import SMOTE

smote=SMOTE() # Synthetic Minority Over-sampling Technique 

X_ovs,y_ovs=smote.fit_resample(X,y)

fig, oversp = plt.subplots()
oversp.pie( y_ovs.value_counts(), autopct='%.2f%%')
oversp.set_title("Over-sampling")
plt.show()
#------------------------------------------------------------
#Train the model  Synthetic Minority Over-sampling Technique (SMOTE)
Xr_train,Xr_test,yr_train,yr_test=train_test_split(X_ovs, y_ovs,test_size=0.2,random_state=42)
from sklearn.linear_model import LogisticRegression
model_lr = LogisticRegression(max_iter=1000)

#Fit the model according to the given training data.
model_lr.fit(Xr_train,yr_train)
y_pred_lr = model_lr.predict(Xr_test)
y_pred_lr[:10]

#Accuracy
model_lr_accuracy = model_lr.score(Xr_test,yr_test)

#classification report for model_lr(SMOTE)
class_report_lr = classification_report(y_pred_lr,yr_test,labels=[0,1])
print(class_report_lr)

#confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(yr_test,y_pred_lr)

#---------------------------------------------------------------------------------------
#Decision Tree classifier
from sklearn.tree import DecisionTreeClassifier
model_dtc=DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=6, min_samples_leaf=8)
model_dtc.fit(Xr_train,yr_train)
y_pred_dtc=model_dtc.predict(Xr_test)
y_pred_dtc[:10]

#tree Accuracy score
model_dtc.score(Xr_test,yr_test)

#tree Confusion matrix
confusion_matrix(yr_test,y_pred_dtc)
#---------------------------------------------------------------------------------------
#Random forest Classifier
from sklearn.ensemble import RandomForestClassifier
model_rfc= RandomForestClassifier(n_estimators=100,random_state = 100,max_depth=6, min_samples_leaf=8,class_weight='balanced')
model_rfc.fit(Xr_train,yr_train)

y_pred_rfc=model_rfc.predict(Xr_test)
y_pred_rfc[:10]
yr_test[:10]

#Accuracy
model_rfc_accuracy = model_rfc.score(Xr_test,yr_test)

cm_rfc = confusion_matrix(yr_test,y_pred_rfc)
#-----------------------------------------------------------------------------------
#AdaBoost
from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(n_estimators=1000)
model_abc = abc.fit(Xr_train,yr_train)

y_pred_abc = model_abc.predict(Xr_test)
y_pred_abc[:10]

#AdaBoost Acccuracy
cllasi_report_abc = classification_report(yr_test,y_pred_abc)

#Confusion matrix for AdaBoost
cm_abc = confusion_matrix(yr_test,y_pred_abc)
cm_abc
#----------------------------------------------------------------------------------
#GradientBoostingClassifer

from sklearn.ensemble import GradientBoostingClassifier
model_gbc = GradientBoostingClassifier()
model_gbc.fit(Xr_train,yr_train)

#prediction Gradient Boosting Classifier
y_pred_gbc=model_gbc.predict(Xr_test)
y_pred_gbc[:10]

print(classification_report(y_pred_gbc,yr_test))

cm_gbc = confusion_matrix(yr_test,y_pred_gbc)
cm_gbc

#----------------------------------------------------------------------------------
#Xgboost
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
model_xgb = XGBClassifier()
model_xgb

#XGb model fit
model_xgb.fit(Xr_train,yr_train)

#Predict XGBC
y_pred_xgb=model_xgb.predict(Xr_test)
y_pred_xgb[:10]

#XGBC Classification report
class_rept_xgb = print(classification_report(y_pred_xgb,yr_test))

#confusion matrix for xgb
cm_xgb=confusion_matrix(yr_test,y_pred_xgb)
#-------------------------------------------------------------------------------------
#GradientBoostingClassifier and adaboost has accuracy i go with Gradientboostingclassifier /finding the best hyperparameter
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import time

# Define your GradientBoostingClassifier and param_dist
model = GradientBoostingClassifier()
param_dist = {
    'learning_rate': [0.1, 0.5, 1.0],
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],  # Example: Adding max_depth parameter
    'min_samples_split': [2, 5, 10]  # Example: Adding min_samples_split parameter
}

# Create RandomizedSearchCV object with fewer iterations
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=5, cv=10, scoring='accuracy', random_state=42)

# Start the timer
start_time = time.time()

# Fit the RandomizedSearchCV object
random_search.fit(Xr_train, yr_train)

# Stop the timer
end_time = time.time()

# Calculate the total time taken
total_time = end_time - start_time

print("RandomizedSearchCV took {:.2f} seconds to complete.".format(total_time))

# Get the best parameters
best_params = random_search.best_params_
print("Best Parameters:", best_params)
#--------------------------------------------------------------------------------------
#final model
from sklearn.ensemble import GradientBoostingClassifier

# Define the best hyperparameters obtained from GridSearchCV
best_params = {
   'n_estimators': 100, 'min_samples_split':5 , 'max_depth': 7, 'learning_rate': 0.1
    
    
 }

# Create Gradient Boosting Classifier with the best hyperparameters
final_gb_classifier = GradientBoostingClassifier(**best_params)

# Train the final model on the entire training data
final_gb_classifier.fit(Xr_train, yr_train)
#------
#Cross Validation by using 
# trained model with tuned hyperparameters
# X_train and y_train are your training data
# cv=10 indicates 10-fold cross-validation
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(final_gb_classifier, Xr_train, yr_train, cv=10, scoring='accuracy')

# Print the cross-validation scores
print("Cross-validation scores:", cv_scores)
print("Mean CV score:", cv_scores.mean())
# Print the cross-validation scores
gradient_acc = print("Cross-validation scores:", cv_scores)
print(gradient_acc)
print("Mean CV score:", cv_scores.mean())

#predict final model
y_pred_gradient=final_gb_classifier.predict(Xr_test)
y_pred_gradient[:10]
#Classsification report
gradient_class_report = print(classification_report(y_pred_gradient,yr_test))

#gradient_accuracy = cv_scores.score(Xr_test, yr_test)

#Confusion matrix
cm_gradient = confusion_matrix(y_pred_gradient,yr_test)

# pickle the file
import os 
import pickle
from sklearn.ensemble import GradientBoostingClassifier

# Change directory if needed
os.chdir(r'D:\Data_Science&AI\Spyder\churn_project')

# Assuming final_gb_classifier is your trained model
# Define and train Gradient Boosting Classifier
best_params = {
    'n_estimators': 100,
    'min_samples_split': 5,
    'max_depth': 7,
    'learning_rate': 0.1
}

final_gb_classifier = GradientBoostingClassifier(**best_params)

# Train the final model on the entire training data (assuming Xr_train and yr_train are defined)
final_gb_classifier.fit(X_train, y_train)

# Dumping the model to a file
with open('final_gb_classifier.pkl', 'wb') as file:
    pickle.dump(final_gb_classifier, file)

# Load the saved model
with open('final_gb_classifier.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

#check accuracy with features

import pickle
import pandas as pd

# Load the saved model from the pickle file
with open('final_gb_classifier.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Prepare your own data for testing
# Create a DataFrame with your feature data
your_features = pd.DataFrame({
    'gender': [1, 0, 0, 0, 0],
    'SeniorCitizen': [0, 0, 0, 0, 0],
    'Partner': [0, 0, 0, 1, 1],
    'Dependents': [0, 0, 0, 0, 1],
    'PhoneService': [1, 0, 1, 1, 1],
    'MultipleLines': [0, 0, 0, 2, 2],
    'InternetService': [1, 0, 1, 1, 0],
    'OnlineSecurity': [0, 0, 0, 2, 2],
    'OnlineBackup': [0, 0, 1, 2, 2],
    'DeviceProtection': [0, 0, 0, 0, 2],
    'TechSupport': [0, 0, 0, 2, 2],
    'StreamingTV': [0, 1, 0, 0, 0],
    'StreamingMovies': [0, 1, 0, 0, 0],
    'Contract': [2, 0, 0, 1, 2],
    'PaperlessBilling': [0, 1, 0, 0, 0],
    'PaymentMethod': [1, 1, 1, 0, 0],
    'MonthlyCharges': [90.407734, 58.273891, 74.379767, 108.55, 64.35],
    'TotalCharges': [707.535237, 3264.466697, 1146.937795, 5610.7, 1558.65],
    'tenure_group': [0, 4, 1, 4, 2]
})

# Make predictions using the loaded model on your own data
predictions = loaded_model.predict(your_features)

# Print the predictions
print("Predictions", predictions)


# Create a DataFrame with actual and predicted values
results = pd.DataFrame({

    'Dropout_actual': yr_test,
    'Dropout_Predicted': y_pred_gradient
    
    
})

results1 = pd.DataFrame({

    'actual_Churn': telco_base_data['Churn'],
    'MonthlyCharges': telco_base_data['MonthlyCharges'],
    'TotalCharges': telco_base_data['TotalCharges'],
    'tenure' :telco_base_data['tenure'],
    
    
})

results1 = pd.DataFrame({
    'gender': telco_base_data['gender'],
    'actual_Churn': telco_base_data['Churn'],
    'MonthlyCharges': telco_base_data['MonthlyCharges'],
    'TotalCharges': telco_base_data['TotalCharges'],
    'tenure':telco_base_data['tenure'],
    'Contract' :telco_base_data['Contract'],
    
})

# Save DataFrame to CSV
results.to_csv('model_predictions.csv', index=False)




trained_telco_data = pd.DataFrame(telco_data)
#predicted_data_set = pd.DataFrame()
Xr_train = pd.DataFrame(Xr_train)

#y_pred = pd.DataFrame(y_test)
#predicted_data_set = pd.DataFrame()
y_pred = pd.DataFrame(y_test)

# Save DataFrame to CSV
results1.to_csv('main_data.csv', index=False)

# Save DataFrame to CSV

trained_telco_data.to_csv('trained_telco_data.csv', index=False)
Xr_train.to_csv('trained_Xr_train.csv', index=False)
y_pred.to_csv('y_pred.csv', index=False)

import os
print(os.getcwd())