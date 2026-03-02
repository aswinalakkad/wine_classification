import streamlit as st
import numpy as np
import pickle

# -----------------------------
# Load Saved Models
# -----------------------------
with open('final_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('lda.pkl', 'rb') as file:
    LDA = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# -----------------------------
# Prediction Function
# -----------------------------
def prediction(input_data):
    scaled_data = scaler.transform(input_data)
    lda_data = LDA.transform(scaled_data)
    pred = model.predict(lda_data)[0]

    if pred == 1:
        return "Wine 1"
    elif pred == 2:
        return "Wine 2"
    else:
        return "Wine 3"


# -----------------------------
# Main App
# -----------------------------
def main():
    st.title("🍷 Wine Classification App")
    st.subheader("Classify wine based on chemical properties")

    st.write("Enter the chemical properties below:")

    # Create two columns for better UI
    col1, col2 = st.columns(2)

    with col1:
        alc = st.number_input("Alcohol", value=None, placeholder="e.g. 13.2")
        mal_acid = st.number_input("Malic Acid", value=None)
        ash = st.number_input("Ash", value=None)
        alc_ash = st.number_input("Alcalinity of Ash", value=None)
        mag = st.number_input("Magnesium", value=None)
        phe = st.number_input("Total Phenols", value=None)
        fla = st.number_input("Flavanoids", value=None)
        nfla = st.number_input("Nonflavanoid Phenols", value=None)
        pro = st.number_input("Proanthocyanins", value=None)
        co_i = st.number_input("Color Intensity", value=None)
        hue = st.number_input("Hue", value=None)
        od = st.number_input("OD280/OD315", value=None)
        proline = st.number_input("Proline", value=None)

    # -----------------------------
    # Predict Button
    # -----------------------------
    if st.button("Predict"):
        inputs = [
            alc, mal_acid, ash, alc_ash, mag,
            phe, fla, nfla, pro, co_i,
            hue, od, proline
        ]

        # Check if any field is empty
        if None in inputs:
            st.warning("⚠️ Please fill all input fields.")
        else:
            try:
                input_array = np.array([inputs], dtype=float)
                result = prediction(input_array)
                st.success(f"Predicted Class: {result}")

            except Exception as e:
                st.error("Something went wrong during prediction.")
                st.write(e)


# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    main()

