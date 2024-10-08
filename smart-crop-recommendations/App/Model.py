import streamlit as st
import joblib
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names, but.*")

# Home page
st.set_page_config(
    page_title="Smart Crop",
    page_icon="logo.webp",
    layout="centered",
)

# Our model
model = None

def load_model():
    global model
    with open("model.pkl", "rb") as f:
        model = joblib.load(f)

def main():

    html_text = """
    <style>
    p {font-size :17px;}
    .block-container {padding: 2rem 1rem 3rem;}
    #MainMenu {visibility: hidden;}
    </style>
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:center;"> SmartCrop: Crop Recommendation</h1>
    </div>
    """
    
    st.markdown(html_text, unsafe_allow_html=True)
   
    st.subheader("Find out the most suitable crop to grow in your Farm")

    # Feature inputs
    N = st.number_input("Nitrogen (ppm)", 1, 100)
    P = st.number_input("Phosphorus (ppm)", 1, 100)
    K = st.number_input("Potassium (ppm)", 1, 100)
    temp = st.number_input("Temperature(°C)", 0.0, 100.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0)
    ph = st.number_input("PH", 0.0, 14.0)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 10000.0)

    feature = np.array([[N, P, K, temp, humidity, ph, rainfall]]).reshape(1, -1)

    # Prediction and display answer
    if st.button("Predict"):
        if model:
            crop = model.predict(feature)
            st.write("## Result")
            st.success(f"{crop[0]} will be the Best For Your Farm This Time.")

            # Create a DataFrame with input features and prediction
            result_df = pd.DataFrame({
                "Feature": [
                    "Nitrogen",
                    "Phosphorus",
                    "Potassium",
                    "Temperature",
                    "Humidity",
                    "PH",
                    "Rainfall",
                    "Predicted Crop"
                ],
                "Entered Value": np.append(feature[0], crop[0])  # Add the predicted crop
            })

            # Display input features
            st.subheader("Input Features:")
            st.write(result_df)
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name='crop_recommendation_results.csv',
                mime='text/csv'
            )

if __name__ == "__main__":
    load_model()
    main()
