import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
import pickle
import streamlit as st

# st.cache_data.clear()

# df = pd.read_csv('phone_usage_cleaned.csv')

st.header("Phone Usage Predictor")

st.subheader('Enter Required Field')

#user Input
#Age	Gender	Location	Phone Brand     Screen Time (hrs/day)	Data Usage (GB/month)	Calls Duration (mins/day)	
# Number of Apps Installed	Social Media Time (hrs/day)	E-commerce Spend (INR/month)	Streaming Time (hrs/day)	
# Gaming Time (hrs/day) 	Monthly Recharge Cost (INR)

Age = st.number_input('Enter Age', min_value = 15, max_value = 60, step = 1)
Gender = st.selectbox('Select Gender', ['Male', 'Female', 'Other'])
Location = st.selectbox('Select Location', ['Mumbai', 'Delhi', 'Ahmedabad', 'Pune', 'Jaipur', 'Lucknow', 'Kolkata',
    'Bangalore', 'Chennai', 'Hyderabad'])
Phone_Brand = st.selectbox('Select Phone Brand', ['Vivo', 'Realme', 'Nokia', 'Samsung', 'Xiaomi', 'Oppo', 'Apple', 'Google Pixel',
    'Motorola', 'OnePlus'])
OS = st.selectbox('Select Phone OS', ['Android', 'iOS'])
screen_time = st.number_input('Enter Screen Time (hrs/day)',min_value=1.0, max_value=12.0, step=0.5)
data_usage = st.number_input('Enter Data Usage (GB/month)', min_value = 1.0, max_value = 60.0, step = 0.5)
call_duration = st.number_input('Enter Calls Duration (mins/day)', min_value = 5, max_value = 300)
num_apps = st.number_input('Enter Number of Apps Installed', min_value = 10, max_value = 200, step = 1)
social_media = st.number_input('Enter Social Media Time (hrs/day)', min_value = 0.5, max_value = 6.0, step = 0.5)
ecomm_spend = st.number_input('Enter Amont spent on E-commerce (INR/month)', min_value = 100, max_value = 10000)
streaming_time = st.number_input('Enter Streaming Time (hrs/day)', min_value = 0.5, max_value = 8.0)
gaming_time = st.number_input('Enter Gaming Time (hrs/day)', min_value = 0, max_value = 5)
recharge_cost  = st.number_input('Enter Monthly Recharge Cost (INR)', min_value = 100, max_value = 2000)

#Create Dictionary of Input Items

user_input_dict = {
    'Age': Age,
    'Gender':Gender,
    'Location':Location,
    'Phone Brand':Phone_Brand,
    'OS':OS,
    'Screen Time (hrs/day)':screen_time,
    'Data Usage (GB/month)':data_usage,
    'Calls Duration (mins/day)':call_duration,
    'Number of Apps Installed':num_apps,
    'Social Media Time (hrs/day)':social_media,
    'E-commerce Spend (INR/month)':ecomm_spend,
    'Streaming Time (hrs/day)':streaming_time,
    'Gaming Time (hrs/day)':gaming_time,
    'Monthly Recharge Cost (INR)':recharge_cost
}

user_input_df = pd.DataFrame([user_input_dict])
user_input_df

#Preprocessing user input_df
# 1. Scaling: MinMaxScaler
# 2. Creating dummies of Object DataType: pd.getdummies()


#1.1 import minmaxscaler pickle file
with open('minmax_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

#1.2 fit_transform
user_input_df[['Age', 'Screen Time (hrs/day)', 'Data Usage (GB/month)',
        'Calls Duration (mins/day)', 'Number of Apps Installed',
        'Social Media Time (hrs/day)', 'E-commerce Spend (INR/month)',
        'Streaming Time (hrs/day)', 'Gaming Time (hrs/day)',
        'Monthly Recharge Cost (INR)']] = scaler.transform(user_input_df[['Age', 'Screen Time (hrs/day)', 'Data Usage (GB/month)',
        'Calls Duration (mins/day)', 'Number of Apps Installed',
        'Social Media Time (hrs/day)', 'E-commerce Spend (INR/month)',
        'Streaming Time (hrs/day)', 'Gaming Time (hrs/day)',
        'Monthly Recharge Cost (INR)']])

# user_input_df
#2.1 - generating dummies

user_input_df = pd.get_dummies(user_input_df, dtype = int)

# user_input_df

#column names used in trained model
model_columns = ['Age', 'Screen Time (hrs/day)', 'Data Usage (GB/month)',
        'Calls Duration (mins/day)', 'Number of Apps Installed',
        'Social Media Time (hrs/day)', 'E-commerce Spend (INR/month)',
        'Streaming Time (hrs/day)', 'Gaming Time (hrs/day)',
        'Monthly Recharge Cost (INR)', 'Gender_Female', 'Gender_Male',
        'Gender_Other', 'Location_Ahmedabad', 'Location_Bangalore',
        'Location_Chennai', 'Location_Delhi', 'Location_Hyderabad',
        'Location_Jaipur', 'Location_Kolkata', 'Location_Lucknow',
        'Location_Mumbai', 'Location_Pune', 'Phone Brand_Apple',
        'Phone Brand_Google Pixel', 'Phone Brand_Motorola', 'Phone Brand_Nokia',
        'Phone Brand_OnePlus', 'Phone Brand_Oppo', 'Phone Brand_Realme',
        'Phone Brand_Samsung', 'Phone Brand_Vivo', 'Phone Brand_Xiaomi',
        'OS_Android', 'OS_iOS']

#creating dataframe with columns used in model for passing into model
model_df = pd.DataFrame(columns = model_columns)

for column in user_input_df:
    if column in model_columns:
        model_df[column] = user_input_df[column]



model_df = model_df.fillna(0)

# model_df

# st.write(model_df.columns)

with open('gbcl.pkl', 'rb') as file:
    model = pickle.load(file)

predict_button = st.button('Predict')
if predict_button:
    prediction = model.predict(model_df)
    st.subheader(*prediction)