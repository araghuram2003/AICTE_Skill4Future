# Water Quality Prediction – Multi-Target Regression

This project focuses on predicting multiple water quality parameters using machine learning techniques, specifically a MultiOutputRegressor wrapped around a RandomForestRegressor. It was developed as part of an internship with **Edunet Foundation** in June 2025.

---
## Overview

Access to clean and safe water is essential for health and environmental sustainability. Predicting water quality parameters accurately helps monitor pollution levels and take timely actions to prevent water contamination.

In this project, the following steps were performed:

- Collected and preprocessed real-world water quality datasets  
- Converted date columns and engineered features such as year and month  
- Applied one-hot encoding to categorical variables (station IDs)  
- Used supervised machine learning for multi-target regression  
- Built a predictive model using MultiOutputRegressor with RandomForestRegressor  
- Evaluated the model’s performance using regression metrics  

---
## Technologies Used

- Python 3.11+ 
- Pandas, NumPy – For data manipulation and numerical operations  
- Scikit-learn – For machine learning model building and evaluation  
- Jupyter Notebook – Interactive coding and experimentation

---

## Predicted Water Quality Parameters

The model predicts various water quality parameters including (but not limited to):

- NH4  
- BOD5 (BSK5)  
- Colloids  
- O2, NO3, NO2, SO4, PO4  
- CL  

---
## Model Performance

The model evaluation was performed using:

- R² Score  
- Mean Squared Error (MSE)  

Overall, the model showed good performance in predicting multiple water quality parameters simultaneously.

---
## How to Run

1. Clone this repository.  
2. Install the required libraries (`pandas`, `numpy`, `scikit-learn`, etc.).  
3. Load the dataset file `PB_All_2000_2021.csv`.  
4. Run the Jupyter Notebook step-by-step to preprocess data, train the model, and evaluate performance.  

---
## Model and Files Saved

- Trained model saved as `pollution_model.pkl`  
- Feature columns saved as `model_columns.pkl`
- model drive link: https://drive.google.com/file/d/1IO1dSxxuYlJyeTlQltoQKyKA55Pw8fE3/view?usp=sharing

These files can be loaded later to make predictions on new data without retraining.

---

## Internship Details

- Internship Provider: Edunet Foundation, Shell and AICTE
- Duration: June 2025 (1 month)  
- Focus: Machine Learning for Environmental monitoring
