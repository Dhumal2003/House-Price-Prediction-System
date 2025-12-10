#python -m streamlit --version  ----- it is used to check version 
#python -m streamlit run app.py ------ it is used to run streamlit 

import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import time


# Load model, scaler, dataset
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")
dataset = joblib.load("preprocessed_indian_house_prices1.pkl")

# ---------- TITLE ----------
st.set_page_config(page_title="House Price Prediction", layout="wide")
st.markdown("<h1 style='text-align:center; color:#F1C40F;'>🏠 House Price Prediction </h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#BDC3C7;'>Fill in the house details on the left and get an estimated price instantly!</p>", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
st.sidebar.header("Enter House Details")
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 10, 2)
lot_area = st.sidebar.number_input("Lot Area (sq ft)", 500, 100000, 5000)
built_year = st.sidebar.slider("Built Year", 1900, 2025, 2000)
grade = st.sidebar.slider("House Grade", 1, 13, 7)

predict = st.sidebar.button("Predict Price")

# ---------- PREDICTION ----------
if predict:
    with st.spinner("🔮 Analyzing your house details..."):
        time.sleep(2) 
        
        inputs = pd.DataFrame({
        "number of bedrooms": [bedrooms],
        "number of bathrooms": [bathrooms],
        "lot area": [lot_area],
        "grade of the house": [grade],
        "Built Year": [built_year],
        "waterfront present": [0],
        "condition of the house": [5],
        "Postal Code": [122003],
        "Lattitude": [52.8645],
        "Longitude": [-114.557],
        "living_area_renov": [bedrooms * 500],
        "lot_area_renov": [lot_area],
        "Number of schools nearby": [2],
        "Distance from the airport": [10],
        "House_Age": [2025 - built_year],
        "Total_Area": [lot_area],
        "number of floors": [1]
    })

    train_cols = [c for c in dataset.columns if c != "Price"]
    inputs = inputs[train_cols]
    input_scale = scaler.transform(inputs)

    prediction = model.predict(input_scale)
    price = int(prediction[0])
    price_lakhs = price / 100000

    col1, col2 = st.columns([1.2, 1])

    with col1:
        img = Image.open("house_image.jpg")
        st.image(img, caption="Your Dream House",  width="stretch")

    with col2:
        with col2:
            st.success("✅ Prediction Complete!")
            st.markdown(
            f"""
            <div style='background:#ECFDF5; padding:30px; border-radius:20px; 
                        text-align:center; width:100%;
                        box-shadow:0 6px 15px rgba(0,0,0,0.25);'>
                <h2 style='color:#117A65;'>🏡 Estimated Price :</h2>
                <p style='font-size:35px; font-weight:bold; color:#E74C3C;'> = ₹ {price:,.0f}  </p>
                <p style='font-size:20px; color:#2C3E50;'>≈ {price_lakhs:.2f} Lakhs</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    # ---------- MESSAGE ----------
    st.markdown(
        """
        <p style='font-size:18px; font-weight:bold; color:#FF8C00; text-align:center; margin-top:25px;'>
            💡 This is an <b>estimated price</b> based on your inputs.  
            Market conditions may affect the actual value.
        </p>
        """,
        unsafe_allow_html=True
    )