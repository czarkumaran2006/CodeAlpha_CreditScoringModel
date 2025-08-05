CREDIT SCORE MODEL

✅ 1. Goal
Build a credit scoring classification model that predicts whether an applicant is a good or bad credit risk.
________________________________________
✅ 2. Steps in Google Colab
Here’s a complete pipeline you can copy into a Google Colab notebook.
________________________________________
📌 Step-by-Step Code:
python
CopyEdit
# STEP 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# STEP 2: Load Dataset (space-separated)
url = '/content/sample_data/german.csv'  # Or upload your own
df = pd.read_csv(url, delim_whitespace=True, header=None)

# STEP 3: Add Column Names (based on German Credit dataset spec)
columns = [
    "Status", "Duration", "CreditHistory", "Purpose", "CreditAmount", "Savings",
    "EmploymentSince", "InstallmentRate", "PersonalStatusSex", "OtherDebtors",
    "ResidenceSince", "Property", "Age", "OtherInstallmentPlans", "Housing",
    "NumberCredits", "Job", "LiablePeople", "Telephone", "ForeignWorker", "Target"
]
df.columns = columns

# STEP 4: Encode Categorical Features
cat_cols = df.select_dtypes(include='object').columns

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# STEP 5: Split Data
X = df.drop("Target", axis=1)
y = df["Target"]

# STEP 6: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 7: Train a Classifier (Random Forest)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# STEP 8: Evaluate
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
________________________________________
🧠 Notes:
•	The Target column values are typically:
o	1 = Good credit
o	2 = Bad credit
You can invert them or make it binary (0/1) if needed.
•	You can upload your own german.csv to Colab via Files > Upload.

