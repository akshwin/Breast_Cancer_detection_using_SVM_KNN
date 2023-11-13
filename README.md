# Breast Cancer Detection using SVM and KNN

Breast cancer is a significant health concern worldwide, emphasizing the need for early detection and accurate diagnosis. This project focuses on developing a breast cancer detection model using two powerful machine learning algorithms: Support Vector Machine (SVM) and k-Nearest Neighbors (KNN). The dataset used is sourced from the UCI Machine Learning Repository, containing various attributes associated with breast cancer tumors.

## Steps Followed

### 1. Data Preprocessing

```python
import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv("breastCancer.csv")

# Handling missing values represented as '?'
df = data.replace("?", np.nan)
df['bare_nucleoli'] = df['bare_nucleoli'].astype('float64')
df['bare_nucleoli'] = df['bare_nucleoli'].fillna(df['bare_nucleoli'].median())
```

### 2. Exploratory Data Analysis (EDA)

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Descriptive statistics
df.describe().T

# Visualizations
sns.pairplot(df, diag_kind='kde')
plt.show()

# Correlation matrix heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', vmax=1, vmin=-1)
plt.title("Correlation between attributes")
plt.show()
```

### 3. Building the Model

```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Split dataset into features (X) and target variable (y)
x = df.drop('class', axis=1)
y = df['class']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# K-Nearest Neighbors (KNN)
KNN = KNeighborsClassifier(n_neighbors=5, weights='distance')
KNN.fit(x_train, y_train)

# Support Vector Machine (SVM)
svc = SVC()
svc.fit(x_train, y_train)
```

### 4. Model Evaluation

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# KNN predictions and classification report
KNN_predict = KNN.predict(x_test)
print("KNN CLASSIFICATION REPORT")
print(classification_report(y_test, KNN_predict))

# Confusion Matrix for KNN
cm_knn = confusion_matrix(y_test, KNN_predict)
sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Blues", xticklabels=['Predict 2', 'Predict 4'], yticklabels=['Actual 2', 'Actual 4'])
plt.show()

# SVM predictions and classification report
svc_predict = svc.predict(x_test)
print("SVC CLASSIFICATION REPORT")
print(classification_report(y_test, svc_predict))

# Confusion Matrix for SVM
cm_svm = confusion_matrix(y_test, svc_predict)
sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Greens", xticklabels=['Predict 2', 'Predict 4'], yticklabels=['Actual 2', 'Actual 4'])
plt.show()
```

## Source

The dataset utilized in this project is obtained from the UCI Machine Learning Repository, a reputable source for machine learning datasets. The implementation of machine learning models and data exploration heavily relied on popular Python libraries such as pandas, numpy, seaborn, scikit-learn, and matplotlib.

## Future Work

- **Hyperparameter Tuning:**
  Fine-tuning the hyperparameters of both the KNN and SVM models could potentially enhance their performance.

- **Additional Algorithms:**
  Exploring alternative machine learning algorithms and ensemble methods for comparison could provide a more comprehensive understanding of the dataset.

## Conclusion

This project showcases the application of machine learning techniques for breast cancer detection, emphasizing their potential contributions to medical research and healthcare. The combination of rigorous data preprocessing, exploratory data analysis, and model development has laid the foundation for accurate and reliable breast cancer detection. The presented code snippets provide a practical guide for implementing and evaluating these models.
