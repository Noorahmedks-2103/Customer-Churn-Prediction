import pandas as pd
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Load training columns from model
feature_names = model.feature_names_in_

# Create empty row with all features
sample_df = pd.DataFrame(columns=feature_names)
sample_df.loc[0] = 0  # fill everything with zero initially

# Manually update required fields
sample_df["tenure"] = 12
sample_df["MonthlyCharges"] = 80.5
sample_df["TotalCharges"] = 960

# Example encoded values:
sample_df["gender_Male"] = 1
sample_df["Partner_Yes"] = 1
sample_df["PhoneService_Yes"] = 1
sample_df["InternetService_Fiber optic"] = 1
sample_df["PaperlessBilling_Yes"] = 1
sample_df["PaymentMethod_Electronic check"] = 1

# Predict
prediction = model.predict(sample_df)[0]

print("Prediction:", "Churn" if prediction == 1 else "Not Churn")
