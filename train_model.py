import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load dataset
data = pd.read_csv('hand_signs_data.csv')  # adjust path if needed

# Rename the last column to 'label'
data.columns.values[-1] = 'label'

print("Renamed Columns:", data.columns.tolist())  # Debug print

# Now split features and labels
X = data.drop('label', axis=1)
y = data['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n✅ Model trained. Classification Report:\n")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, 'models/hand_sign_model.pkl')
print("\n📁 Model saved to: models/hand_sign_model.pkl")
