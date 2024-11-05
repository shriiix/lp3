import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load your dataset
data = pd.read_csv('emails.csv')

# Inspect the data types and check for unexpected types
print(data.dtypes)

# Drop non-numeric columns, keeping only numeric ones for X
X = data.drop(columns=['Prediction'])  # All columns except 'Prediction'
X = X.select_dtypes(include=['int64', 'float64'])  # Keep only integer and float columns

# Ensure y is numeric (0 and 1)
y = data['Prediction'].astype(int)  # Convert to integer type

# Check for NaN values in X and y
print("NaN values in X:", X.isna().sum().sum())
print("NaN values in y:", y.isna().sum())

# Drop rows with NaN values in features
X = X.dropna()
y = y[X.index]  # Ensure y aligns with the features after dropping NaN values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y.to_numpy(), test_size=0.2, random_state=42)

# K-Nearest Neighbors Classification
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Support Vector Machine Classification
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# Performance Analysis
print("KNN Classification Report:")
print(classification_report(y_test, y_pred_knn))
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))

print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
