#Streamlit
import streamlit as st
import pandas as pd
import pickle
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
# Load the trained model
#model = joblib.load(r'final_gb_classifier.pkl')
data_set = pd.read_csv(r"Telco-Customer-Churn.csv")
# Streamlit UI

st.title("Customer Retention Prediction")

st.write("<h5 style='text-align: left; color: blue;'>Understand and predict Customer Retention to improve retention rates and reduce lost revenue </h5>", unsafe_allow_html=True)
st.markdown("""
Using `GradientBoostingClassifier` to predict the likelihood of Customer Retention based on the following features:
- **gender**
- **SeniorCitizen**
- **Partner**
- **Dependents**
- **tenure**
- **PhoneService**
- **MultipleLines**
- **InternetService**
- **OnlineSecurity**
- **OnlineBackup**
- **DeviceProtection**
- **TechSupport**
- **StreamingTV**
- **StreamingMovies**
- **Contract**
- **PaperlessBilling**
- **PaymentMethod**
- **MonthlyCharges**
- **TotalCharges**
""")

# Function to preprocess input data
def preprocess_input(data):
    # Convert input data to DataFrame
    df = pd.DataFrame(data, index=[0])
     # Convert categorical variables to numeric
    df['InternetService'] = df['InternetService'].map({'DSL': 0, 'Fiber optic': 1, 'No': 2})
    df['Contract'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    df['PaymentMethod'] = df['PaymentMethod'].map({'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3})
    # Return preprocessed DataFrame
    return df
# Collect user inputs
gender = st.radio("Gender", [0, 1])
senior_citizen = st.radio("Senior Citizen", [0, 1])
partner = st.radio("Partner", [0, 1])
dependents = st.radio("Dependents", [0, 1])
phone_service = st.radio("Phone Service", [0, 1])
multiple_lines = st.radio("Multiple Lines", [0, 1])
internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
online_security = st.radio("Online Security", [0, 1, 2])
online_backup = st.radio("Online Backup", [0, 1, 2])
device_protection = st.radio("Device Protection", [0, 1, 2])
tech_support = st.radio("Tech Support", [0, 1, 2])
streaming_tv = st.radio("Streaming TV", [0, 1])
streaming_movies = st.radio("Streaming Movies", [0, 1])
contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.radio("Paperless Billing", [0, 1])
payment_method = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
monthly_charges = st.number_input("Monthly Charges", value=0.0)
total_charges = st.number_input("Total Charges", value=0.0)
tenure_group = st.number_input("Tenure Group", value=0)
#====================================================================================================

# Make prediction
if st.button("Predict"):
    # Create dictionary from user inputs
    user_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'tenure_group': tenure_group,

    }

    
    # Preprocess input data
    processed_data = preprocess_input(user_data)
    
    # Make prediction
    prediction = model.predict(processed_data)
    
    # Display prediction result
    if prediction[0] == 1:
        st.success("The customer is likely to Retentine.", icon="❎")
    else:
        #st.write("**The customer is likely to stay.**")
        st.success('The customer is likely to stay.', icon="✅")
        st.balloons()
    
        
        
#st.write('**Original Dataset**')
Predicted_data = st.write("<h5 style='text-align: left; color: purple;'>Original Dataset</h5>", unsafe_allow_html=True)
st.write(pd.DataFrame(data_set))
#st.write("<h5 style='text-align: left; color: blue;'> </h5>", unsafe_allow_html=True)

#=========================================================================================
trained_dropout_data= pd.read_csv(r'trained_telco_data.csv')
trained_Xr_train= pd.read_csv(r"trained_Xr_train.csv")
y_pred= pd.read_csv(r"y_pred.csv")
#============================================================================
#Predicted_data = st.write('**Trained DataSet**')
Predicted_data = st.write("<h5 style='text-align: left; color: purple;'>Trained DataSet </h5>", unsafe_allow_html=True)
st.write(pd.DataFrame(trained_dropout_data))

st.write("<h2 style='text-align: left; color:Green;'>Visualization </h2>", unsafe_allow_html=True)
st.write("<h5 style='text-align: left; color: red;'>Scatter_Plot(TotalCharges Vs MonthlyCharges) </h5>", unsafe_allow_html=True)
#st.write("**Scatter_Plot(TotalCharges Vs MonthlyCharges)**")
# Create DataFrame with the correct structure
chart_Dropout1 = pd.DataFrame(trained_Xr_train)
#=================================================================================
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='TotalCharges', y='MonthlyCharges', data=chart_Dropout1, ax=ax, color='#803df5')
sns.regplot(x='TotalCharges', y='MonthlyCharges', data=chart_Dropout1, scatter=False, ax=ax, color='red')
st.pyplot(fig)
#=====================================================================================
import seaborn as sns
fig, ax = plt.subplots(figsize=(10, 6))
st.write("<h5 style='text-align: left; color: red;'>Box_Plot(Tenure_group Vs MonthlyCharges) </h5>", unsafe_allow_html=True)
sns.boxplot(x='tenure_group', y='MonthlyCharges', data=trained_dropout_data, ax=ax)
st.pyplot(fig)
#===========================================================
st.write("<h5 style='text-align: left; color: red;'>Bar_Chart(Tenure Vs MonthlyCharges) </h5>", unsafe_allow_html=True)
#st.write("**Bar_Chart(Tenure Vs MonthlyCharges)**")
groudata = trained_dropout_data.groupby('tenure_group')['MonthlyCharges'].mean().reset_index()
import altair as alt
# Create the Altair bar chart
bar_chart = alt.Chart(groudata).mark_bar().encode(x=alt.X('tenure_group', title='Tenure Group'), y=alt.Y('MonthlyCharges', title='Average Monthly Charges'))
# Display the chart with Streamlit
st.altair_chart(bar_chart, use_container_width=True)

#st.scatter_chart(trained_dropout_data, y=telco_data['Churn'], x=telco_data['TotalCharges'],width=700,height=700,color='#21c354')
#==========================================================================================
corr = trained_dropout_data.corr()
print(corr)



# Display correlation matrix as a bar chart
#st.title("Churn Analysis Correlation")
st.write("<h4 style='text-align: left; color: red;'>Churn Analysis Correlation</h4>", unsafe_allow_html=True)
st.write("<h5 style='text-align: left; color: blue;'>Correlation of all predictors with 'Churn'</h5>", unsafe_allow_html=True)
#st.write("**Correlation of all predictors with 'Churn'**")
fig, ax = plt.subplots(figsize=(20, 8))
trained_dropout_data.corr()['Churn'].sort_values(ascending=False).plot(kind='bar', ax=ax)
st.pyplot(fig)

# Display heatmap of the correlation matrix
#st.write("**Correlation Matrix Heatmap**")
st.write("<h4 style='text-align: left; color: red;'>Correlation Matrix Heatmap</h4>", unsafe_allow_html=True)
heatmap = sns.heatmap(corr, cmap='Set1', annot=True, square=True)
#st.pyplot(fig)
#plt.figure(figsize=(20,8))
#telco_data.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')

#HeatMap
fig = plt.figure(figsize=(25,25))
sns.heatmap(corr,cmap='Set1',annot=True, square=True,linewidths=2,cbar=True,fmt='.2g')
st.pyplot(fig)

#=================================================

#=======================================================

#streamlit run Customer_Retention_Prediction_App.py
