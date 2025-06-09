import streamlit as st



pg = st.navigation([
    st.Page("app.py", title="Prediction App", icon="🔢"),
    st.Page("phone_plots.py", title = 'EDA', icon = "📉"),
])
pg.run()