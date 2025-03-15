import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the CSV file
df = pd.read_csv("Ramya_Education_Uptor_Final_Project.csv")

# Check the first few rows to ensure data is loaded correctly
print(df.head())

# Supervised Learning: Random Forest Classifier
X = df[['Study Hours', 'Previous Exam Score']]
y = df['Pass/Fail']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict on the test set
y_pred = rf.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of Random Forest Classifier: {accuracy*100:.2f}%")

# Unsupervised Learning: K-Means Clustering
# Standardize the features before applying KMeans
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans clustering (assuming 2 clusters for simplicity)
kmeans = KMeans(n_clusters=2, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# View the resulting clusters
print(df[['Study Hours', 'Previous Exam Score', 'cluster']])

# Plot clusters
plt.scatter(df['Study Hours'], df['Previous Exam Score'], c=df['cluster'], cmap='viridis')
plt.xlabel('Study Hours')
plt.ylabel('Previous Scores')
plt.title('K-Means Clustering of Students')
plt.show()
