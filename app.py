import streamlit as st
import pickle
import numpy as np

with open('mymodel.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("My Machine Learning Web App 🚀")

age = st.number_input("Enter your Age:", min_value=1, max_value=100, value=25)
salary = st.number_input("Enter your Monthly Salary:", min_value=0, value=30000)

if st.button("Predict"):
    features = np.array([[age, salary]])
    prediction = model.predict(features)
    st.success(f"Prediction Result: {prediction[0]}")