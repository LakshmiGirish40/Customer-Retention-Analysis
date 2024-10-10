# Customer-Retention-Analysis
Project Title: Predicting Customer Retention 

📊 𝗖𝘂𝘀𝘁𝗼𝗺𝗲𝗿 𝗥𝗲𝘁𝗲𝗻𝘁𝗶𝗼𝗻 𝗔𝗻𝗮𝗹𝘆𝘀𝗶𝘀 𝗨𝘀𝗶𝗻𝗴 𝗖𝗹𝗮𝘀𝘀𝗶𝗳𝗶𝗰𝗮𝘁𝗶𝗼𝗻 & 𝗟𝗼𝗴𝗶𝘀𝘁𝗶𝗰 𝗥𝗲𝗴𝗿𝗲𝘀𝘀𝗶𝗼𝗻
🎯 𝗢𝗯𝗷𝗲𝗰𝘁𝗶𝘃𝗲:
The project aims to 𝗽𝗿𝗲𝗱𝗶𝗰𝘁 𝗰𝘂𝘀𝘁𝗼𝗺𝗲𝗿 𝗰𝗵𝘂𝗿𝗻 𝗳𝗼𝗿 𝗮 𝘁𝗲𝗹𝗲𝗰𝗼𝗺 𝗰𝗼𝗺𝗽𝗮𝗻𝘆 by analyzing various customer attributes and service usage patterns. This will help the company reduce churn rates and enhance customer retention strategies, improving overall profitability.

🔑 𝗞𝗲𝘆 𝗙𝗲𝗮𝘁𝘂𝗿𝗲𝘀:
👤 𝗖𝘂𝘀𝘁𝗼𝗺𝗲𝗿 𝗗𝗲𝗺𝗼𝗴𝗿𝗮𝗽𝗵𝗶𝗰𝘀:
• 𝗴𝗲𝗻𝗱𝗲𝗿: Gender of the customer (Male/Female).
• 𝗦𝗲𝗻𝗶𝗼𝗿𝗖𝗶𝘁𝗶𝘇𝗲𝗻: Whether the customer is a senior citizen (1 for Yes, 0 for No).
• 𝗣𝗮𝗿𝘁𝗻𝗲𝗿: Whether the customer has a partner (Yes/No).
• 𝗗𝗲𝗽𝗲𝗻𝗱𝗲𝗻𝘁𝘀: Whether the customer has dependents (Yes/No)

📞 𝗦𝗲𝗿𝘃𝗶𝗰𝗲 𝗜𝗻𝗳𝗼𝗿𝗺𝗮𝘁𝗶𝗼𝗻:
• 𝗣𝗵𝗼𝗻𝗲𝗦𝗲𝗿𝘃𝗶𝗰𝗲: Whether the customer has phone service (Yes/No).
• 𝗠𝘂𝗹𝘁𝗶𝗽𝗹𝗲𝗟𝗶𝗻𝗲𝘀: Whether the customer has multiple lines (Yes/No).
• 𝗜𝗻𝘁𝗲𝗿𝗻𝗲𝘁𝗦𝗲𝗿𝘃𝗶𝗰𝗲: Type of internet service (DSL/Fiber optic/No).
• 𝗢𝗻𝗹𝗶𝗻𝗲𝗦𝗲𝗰𝘂𝗿𝗶𝘁𝘆: Online security service (Yes/No/No internet service).
• 𝗢𝗻𝗹𝗶𝗻𝗲𝗕𝗮𝗰𝗸𝘂𝗽: Online backup (Yes/No/No internet service).
• 𝗗𝗲𝘃𝗶𝗰𝗲𝗣𝗿𝗼𝘁𝗲𝗰𝘁𝗶𝗼𝗻: Device protection (Yes/No/No internet service).
• 𝗧𝗲𝗰𝗵𝗦𝘂𝗽𝗽𝗼𝗿𝘁: Tech support availability (Yes/No/No internet service).• 𝗦𝘁𝗿𝗲𝗮𝗺𝗶𝗻𝗴𝗧𝗩: Streaming TV subscription (Yes/No/No internet service).
• 𝗦𝘁𝗿𝗲𝗮𝗺𝗶𝗻𝗴𝗠𝗼𝘃𝗶𝗲𝘀: Streaming movies subscription (Yes/No/No internet service).

📜 𝗖𝗼𝗻𝘁𝗿𝗮𝗰𝘁 𝗜𝗻𝗳𝗼𝗿𝗺𝗮𝘁𝗶𝗼𝗻:
• 𝗖𝗼𝗻𝘁𝗿𝗮𝗰𝘁: Type of contract (Month-to-month, One year, Two years).
• 𝗣𝗮𝗽𝗲𝗿𝗹𝗲𝘀𝘀𝗕𝗶𝗹𝗹𝗶𝗻𝗴: Whether the customer uses paperless billing (Yes/No).
• 𝗣𝗮𝘆𝗺𝗲𝗻𝘁𝗠𝗲𝘁𝗵𝗼𝗱: Payment method (Electronic check, Mailed check, Bank transfer, Credit card).
• 𝗠𝗼𝗻𝘁𝗵𝗹𝘆𝗖𝗵𝗮𝗿𝗴𝗲𝘀: Monthly charges for the customer.
• 𝗧𝗼𝘁𝗮𝗹𝗖𝗵𝗮𝗿𝗴𝗲𝘀: Total charges for the customer.
• 𝘁𝗲𝗻𝘂𝗿𝗲_𝗴𝗿𝗼𝘂𝗽: Grouped tenure based on how long the customer has been with the company.
🎯 𝗧𝗮𝗿𝗴𝗲𝘁 𝗩𝗮𝗿𝗶𝗮𝗯𝗹𝗲:
𝗖𝗵𝘂𝗿𝗻: Whether the customer churned (Yes/No).

🔧 Data Processing:
🧼 Data Cleaning:
• Handle missing values in features such as       TotalCharges.
•  Encode categorical variables using one-hot or label encoding.
•  Normalize continuous features like MonthlyCharges and  TotalCharges.
🏗️ 𝗙𝗲𝗮𝘁𝘂𝗿𝗲 𝗘𝗻𝗴𝗶𝗻𝗲𝗲𝗿𝗶𝗻𝗴:
 • Create tenure_group to categorize customers based on their time with the company.
• Investigate the interactions between service features (e.g., customers 
• with multiple lines and internet services) to see how they influence churn.
✂️ Data Splitting:
Split the dataset into training (80%) and testing (20%) sets for model building and evaluation.

⚙️ Model Implementation:
🔍 Classification Models:
	• Implement various classification algorithms like Decision Trees, Random Forest, and Logistic Regression to predict churn.
	• Use GridSearchCV to perform hyperparameter tuning and optimize model performance.
📉 Logistic Regression:
Build a logistic regression model to determine the probability of churn and interpret the coefficients to understand the key drivers of customer churn.
📊 Model Evaluation:
Assess model performance using metrics like:
	• Accuracy
	• Precision
	• Recall
	• F1-Score
	• AUC-ROC
Identify the best model based on these metrics for final deployment.

🚀 Model Deployment:
🌐 Streamlit Application:
	• Deploy the model as an interactive web application using Streamlit.
	• The app will allow users to:
	• Input customer details.
	• Predict the probability of churn.
	• Visualize key insights through charts and graphs.
	• Enable batch predictions by uploading new customer data for real-time churn predictions.

•🏆 Conclusion:
This project effectively uses classification techniques and logistic regression to predict customer churn. With a user-friendly Streamlit interface, the deployed model provides telecom companies with actionable insights to retain their customers and reduce churn, enhancing overall business performance. 🚀

https://github.com/LakshmiGirish40/Customer_Retention_Prediction_Analysis_App.git
![image](https://github.com/user-attachments/assets/360bfb7d-7379-42da-afd9-1f1e1f986d7b)
