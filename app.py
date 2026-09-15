import streamlit as st
import numpy as np
import pickle

with open('house_model.pkl','rb') as file:
    model_data = pickle.load(file)

w = model_data['weights']
b = model_data['bias']
mu = model_data['mu']
sigma = model_data['sigma']

st.title("🏡 AI Real Estate Price Predictor")
st.write("Enter the house details below to get an instant valuation.")

area = st.number_input("Area (Square Feet)", min_value=1000, max_value=20000, value=5000, step=100)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
stories = st.number_input("Number of Stories", min_value=1, max_value=5, value=2)
parking = st.number_input("Parking Spots", min_value=0, max_value=5, value=1)
ac_text = st.selectbox("Air Conditioning", ["Yes", "No"])

# 3. Process the Inputs
if st.button("Predict Price"):
    # Convert AC text back to 1 or 0
    ac = 1 if ac_text == "Yes" else 0
    
    # Create the feature array in the EXACT order you trained them:
    # ['area', 'bathrooms', 'airconditioning', 'stories', 'parking', 'bedrooms']
    new_house = np.array([area, bathrooms, ac, stories, parking, bedrooms])
    
    # Scale the inputs using the training mean and std dev
    new_house_scaled = (new_house - mu) / sigma
    
    # Calculate the prediction
    predicted_price_scaled = np.dot(new_house_scaled, w) + b
    actual_price = predicted_price_scaled * 100000
    
    # Display the result
    st.success(f"Estimated House Price: ${actual_price:,.2f}")