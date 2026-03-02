

# 🍷 Wine Classification App

A machine learning web application that classifies wines into one of three categories based on their chemical properties.

## Overview

This app uses a supervised classification model with LDA (Linear Discriminant Analysis) for dimensionality reduction to predict wine class from 13 chemical features. It's trained on the classic **UCI Wine Dataset**.

## Live Demo

🔗 https://wine-classification-01.streamlit.app/#wine-classification-app

## Features

- Classifies wine into **Wine 1**, **Wine 2**, or **Wine 3**
- Input validation — warns if any field is left empty
- Clean, interactive UI built with Streamlit

## Input Features

| Feature | Description |
|---|---|
| Alcohol | Alcohol percentage |
| Malic Acid | Malic acid content |
| Ash | Total ash content |
| Alcalinity of Ash | Alkalinity level of ash |
| Magnesium | Magnesium concentration |
| Total Phenols | Total phenolic compounds |
| Flavanoids | Flavanoid content |
| Nonflavanoid Phenols | Non-flavanoid phenolic compounds |
| Proanthocyanins | Proanthocyanin concentration |
| Color Intensity | Wine color intensity |
| Hue | Hue value |
| OD280/OD315 | OD ratio of diluted wines |
| Proline | Proline concentration |

## Tech Stack

- **Frontend:** Streamlit
- **ML Pipeline:** Scikit-learn (Scaler → LDA → Classifier)
- **Language:** Python

## Setup & Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-folder>

# Install dependencies
pip install streamlit numpy scikit-learn

# Run the app
streamlit run app.py
```

> **Note:** Ensure `final_model.pkl`, `lda.pkl`, and `scaler.pkl` are present in the project root.

## Project Structure

```
├── app.py               # Main Streamlit application
├── final_model.pkl      # Trained classification model
├── lda.pkl              # Fitted LDA transformer
├── scaler.pkl           # Fitted data scaler
└── README.md
```

## Model Pipeline

1. Raw inputs are **standardized** using a pre-fitted `StandardScaler`
2. Dimensionality is reduced using **LDA**
3. The classifier predicts the **wine class (1, 2, or 3)**
