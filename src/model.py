import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import pickle

# ================================
# 📂 Load Dataset
# ================================
df = pd.read_csv("Customer_Churn.csv")

# ================================
# 🔧 Convert TotalCharges to numeric
# ================================
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())   # updated to avoid warning

# ================================
# 🎯 Encode Output Column
# ================================
label_encoder = LabelEncoder()
df["Churn"] = label_encoder.fit_transform(df["Churn"])

# ================================
# 🔠 One-hot Encode Categorical Variables
# ================================
df = pd.get_dummies(df, drop_first=True)

# ================================
# ✂ Split Features & Target
# ================================
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ================================
# 🤖 Train Model
# ================================
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ================================
# 🎓 Evaluate Model
# ================================
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred))

# ================================
# 💾 Save Model
# ================================
pickle.dump(model, open("model.pkl", "wb"))
print("Model saved successfully as model.pkl")
