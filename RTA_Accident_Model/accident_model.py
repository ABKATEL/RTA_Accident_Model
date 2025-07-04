# accident_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib

# Sample data embedded directly
data = {
    'Accident_severity': [2, 1, 3, 2, 1, 3, 2, 1, 3, 2],
    'Number_of_vehicles_involved': [2, 3, 4, 2, 1, 3, 2, 3, 4, 2],
    'Number_of_casualties': [1, 2, 3, 1, 0, 2, 1, 2, 3, 1],
    'Weather_conditions': ['Clear', 'Rain', 'Clear', 'Rain', 'Fog', 'Clear', 'Rain', 'Clear', 'Fog', 'Rain'],
    'Light_conditions': ['Daylight', 'Night', 'Night', 'Daylight', 'Night', 'Daylight', 'Night', 'Daylight', 'Night', 'Daylight'],
    'Road_surface_conditions': ['Dry', 'Wet', 'Dry', 'Wet', 'Wet', 'Dry', 'Wet', 'Dry', 'Wet', 'Dry']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert categorical features to numeric
df = pd.get_dummies(df, drop_first=True)

# Split into features and target
X = df.drop('Accident_severity', axis=1)
y = df['Accident_severity']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))

# Save the model
joblib.dump(model, 'accident_severity_model.pkl')
print("✅ Model saved successfully!")

# Sample prediction using hypothetical data
sample_input = X.iloc[[0]].copy()
sample_input.iloc[0] = [2] * len(X.columns)
loaded_model = joblib.load('accident_severity_model.pkl')
prediction = loaded_model.predict(sample_input)
print("📌 Predicted Accident Severity (sample):", prediction)
