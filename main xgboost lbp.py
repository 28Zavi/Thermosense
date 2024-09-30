import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Define CSV file path
csv_path = r"C:\Users\Parth\Desktop\lbp_features_splitted.csv"

# Read data from CSV
data = pd.read_csv(csv_path)

# Separate features and target variable
features = data.drop("class_label", axis=1)  # All columns except "class_label"
target = data["class_label"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Label encode the target variable for training and testing sets
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Create an XGBoost classifier
clf = XGBClassifier(n_estimators=1000, max_depth=4, learning_rate=0.1, random_state=42)

# Train the model with encoded labels on training data
clf.fit(X_train, y_train_encoded)

# Make predictions on the test set
y_pred_encoded = clf.predict(X_test)

# Decode the predicted labels back to original class labels for testing set
y_pred = label_encoder.inverse_transform(y_pred_encoded)

# Evaluate model performance on testing data
accuracy_test = accuracy_score(y_test, y_pred)
precision_test = precision_score(y_test, y_pred, average='weighted')
recall_test = recall_score(y_test, y_pred, average='weighted')
f1_test = f1_score(y_test, y_pred, average='weighted')

"""print("XGBoost Classifier Performance on Training Data:")
print("Accuracy (Training):", accuracy_score(y_train, label_encoder.inverse_transform(clf.predict(X_train))))
print("Precision (Training):", precision_score(y_train, label_encoder.inverse_transform(clf.predict(X_train)), average='weighted'))
print("Recall (Training):", recall_score(y_train, label_encoder.inverse_transform(clf.predict(X_train)), average='weighted'))
print("F1-score (Training):", f1_score(y_train, label_encoder.inverse_transform(clf.predict(X_train)), average='weighted'))
"""
print("\nXGBoost Classifier Performance on Testing Data:")
print("Accuracy (Testing):", accuracy_test)
print("Precision (Testing):", precision_test)
print("Recall (Testing):", recall_test)
print("F1-score (Testing):", f1_test)

# Save the model
import joblib

# Save the XGBoost model
joblib.dump(clf, "xgboost_model_augmented.pkl")

print("\nXGBoost Model saved as xgboost_model_augmented.pkl")
