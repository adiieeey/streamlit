import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

filename = 'passengers.pkl'
with open(filename, 'rb') as file:
    loaded_model = pickle.load(file)

st.title("survival prediction App")
st.subheader("Please ente your data:")

df = pd.read_csv('exam_x')
columns_list = df.columns.to_list()

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    df_preprocessed=uploaded_file.reindex(columns=columns_list,fill_value=0)
    prediction = loaded_model.predict(df_preprocessed)

    st.subheader('passengers survived:')
    st.write(prediction)
