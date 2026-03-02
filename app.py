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
    st.subheader("Classify wine based on its chemical properties")

    st.write("Please enter all the values below:")

    # -----------------------------
    # Input Fields (Single Column)
    # -----------------------------

    alc = st.number_input("Alcohol", value=None)
    st.caption("Alcohol percentage present in wine.")

    mal_acid = st.number_input("Malic Acid", value=None)
    st.caption("Amount of malic acid in the wine.")

    ash = st.number_input("Ash", value=None)
    st.caption("Total ash content of wine.")

    alc_ash = st.number_input("Alcalinity of Ash", value=None)
    st.caption("Alkalinity level of the ash.")

    mag = st.number_input("Magnesium", value=None)
    st.caption("Magnesium concentration.")

    phe = st.number_input("Total Phenols", value=None)
    st.caption("Total phenolic compounds present.")

    fla = st.number_input("Flavanoids", value=None)
    st.caption("Amount of flavanoids in wine.")

    nfla = st.number_input("Nonflavanoid Phenols", value=None)
    st.caption("Non-flavanoid phenolic compounds.")

    pro = st.number_input("Proanthocyanins", value=None)
    st.caption("Proanthocyanin concentration.")

    co_i = st.number_input("Color Intensity", value=None)
    st.caption("Intensity of wine color.")

    hue = st.number_input("Hue", value=None)
    st.caption("Hue value of the wine.")

    od = st.number_input("OD280/OD315", value=None)
    st.caption("OD280/OD315 ratio of diluted wine.")

    proline = st.number_input("Proline", value=None)
    st.caption("Proline concentration in wine.")

    # -----------------------------
    # Prediction Button
    # -----------------------------
    if st.button("Predict"):
        inputs = [
            alc, mal_acid, ash, alc_ash, mag,
            phe, fla, nfla, pro, co_i,
            hue, od, proline
        ]

        if None in inputs:
            st.warning("⚠️ Please fill all fields before predicting.")
        else:
            try:
                input_array = np.array([inputs], dtype=float)
                result = prediction(input_array)
                st.success(f"Predicted Class: {result}")

            except Exception as e:
                st.error("Error during prediction.")
                st.write(e)


# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    main()
