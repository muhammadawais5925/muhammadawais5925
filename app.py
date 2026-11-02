import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

# Keep random_forest_model.pkl in the same folder as this app.py
MODEL_PATH = Path(__file__).with_name("random_forest_model.pkl")

st.set_page_config(page_title="Product Category Predictor", page_icon="🛍️")
st.title("Product Category Predictor")
st.write("Enter product details to predict its category.")

@st.cache_resource
def load_model():
    # Load the trained model saved as a pickle file.
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)

if not MODEL_PATH.exists():
    st.error("Model file not found. Upload random_forest_model.pkl beside app.py.")
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error(f"Model load nahi ho saka: {e}")
    st.info("Ensure this pickle was created with a compatible scikit-learn version.")
    st.stop()

with st.form("product_form"):
    price = st.number_input("Price", min_value=0.0, value=100.0, step=1.0)
    quantity = st.number_input("Quantity", min_value=0, value=1, step=1)
    rating = st.number_input("Rating", min_value=0.0, max_value=5.0, value=4.0, step=0.1)
    revenue = st.number_input("Revenue", min_value=0.0, value=100.0, step=1.0)
    submitted = st.form_submit_button("Predict category")

if submitted:
    # The model in this project was trained with these four feature names.
    input_df = pd.DataFrame(
        [[price, quantity, rating, revenue]],
        columns=["Price", "Quantity", "Rating", "Revenue"],
    )
    try:
        prediction = model.predict(input_df)
        st.success(f"Predicted category: {prediction[0]}")
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            if hasattr(model, "classes_"):
                result = pd.DataFrame({
                    "Category": model.classes_,
                    "Probability": probabilities,
                }).sort_values("Probability", ascending=False)
                st.subheader("Prediction confidence")
                st.dataframe(result, use_container_width=True)
    except Exception as e:
        st.error(f"Prediction nahi ho saki: {e}")
        st.caption("Model ke training features app ke input fields se match hone chahiye.")
