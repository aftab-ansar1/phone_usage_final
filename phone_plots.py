# import matplotlib.pyplot as plt
# import seaborn as sns 
import pandas as pd
import numpy as np
import streamlit as st

df=pd.read_csv('phone_usage_cleaned.csv')


st.header('Exploratory Data Analysis of Phone Usage')
obj_Group = ['Gender', 'Location', 'Phone Brand', 'OS', 'Primary Use']
num_group = ['Age',
    'Screen Time (hrs/day)',
    'Data Usage (GB/month)',
    'Calls Duration (mins/day)',
    'Number of Apps Installed',
    'Social Media Time (hrs/day)',
    'E-commerce Spend (INR/month)',
    'Streaming Time (hrs/day)',
    'Gaming Time (hrs/day)',
    'Monthly Recharge Cost (INR)']

select_obj = st.selectbox('Select Groups to Check the Usage Data', obj_Group)
select_data = st.multiselect('Select Data to Plot', num_group  )
generate_plot = st.button('Enter')
if generate_plot:
    for x in select_data:
        df1 = df.groupby(select_obj)[x].mean()
        st.bar_chart(df1, x_label=  x)

cat_plot_select = st.selectbox('Select the Category to Check The Primary Useage',['Gender', 'Location', 'Phone Brand', 'OS'] )

df2 = pd.DataFrame(df.groupby(cat_plot_select)['Primary Use'].value_counts(normalize=True)).reset_index()

# genrate_graph = st.button('Enter ')
# if genrate_graph:
st.bar_chart(df2, x = cat_plot_select, y = 'proportion', color = 'Primary Use',  stack = True, horizontal = True)