import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define CSV file path
csv_path = r"C:\Users\Parth\Desktop\glcm_features_without_aug.csv"

# Read data from CSV
data = pd.read_csv(csv_path)

# Separate features and target variable
features = data.drop("class_label", axis=1)  # All columns except "class_label"
target = data["class_label"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create a k-Nearest Neighbors classifier
clf = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors (k) as needed

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("k-Nearest Neighbors (k-NN) Classifier Performance:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Save the model
import joblib

# Save the k-NN model
joblib.dump(clf, "knn_model.pkl")

print("k-NN Model saved as knn_model.pkl")
