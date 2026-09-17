import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load Data & Set Index
df = pd.read_csv('TelcoCustomerChurn.csv', on_bad_lines='skip')
if 'CustomerID' in df.columns:
    df.set_index('CustomerID', inplace=True)

# 2. Drop irrelevant metadata
cols_to_drop = ['ZipCode', 'Latitude', 'Longitude', 'ChurnCategory', 
                'ChurnReason', 'ChurnScore', 'CustomerStatus', 'CLTV']
df.drop(columns=cols_to_drop, errors='ignore', inplace=True)

# 3. Handle Binary Encoding
pd.set_option('future.no_silent_downcasting', True)
binary_cols = [
    'Married', 'Dependents', 'ReferredaFriend', 'MultipleLines', 'InternetService', 
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtectionPlan', 'PremiumTechSupport', 
    'PaperlessBilling', 'StreamingTV', 'StreamingMovies', 'StreamingMusic', 
    'UnlimitedData', 'ChurnLabel'
]
# Intersect with existing columns to avoid KeyErrors
existing_binary = [col for col in binary_cols if col in df.columns]
df[existing_binary] = df[existing_binary].replace({'Yes': 1, 'No': 0})

# 4. Clean Numeric Types Properly (Handling blanks and overflow)
if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce').fillna(0)

# Cast integers safely (use int32 for Population to avoid overflow)
int_cols_small = ['Age', 'NumberofDependents', 'Number_of_Referrals', 'TenureinMonths', 
                  'AvgMonthlyGBDownload', 'TotalExtraDataCharges', 'SatisfactionScore']
existing_int_small = [col for col in int_cols_small if col in df.columns]
df[existing_int_small] = df[existing_int_small].astype('int8')

if 'Population' in df.columns:
    df['Population'] = df['Population'].astype('int32')

float_cols = ['AvgMonthlyLongDistanceCharges', 'MonthlyCharge', 'TotalCharges', 
              'TotalRefunds', 'TotalLongDistanceCharges', 'TotalRevenue']
existing_floats = [col for col in float_cols if col in df.columns]
df[existing_floats] = df[existing_floats].astype('float32')

# 5. One-Hot Encoding
df_final = pd.get_dummies(df, drop_first=True)

# 6. Separate Features & Target
# Adjust target column dynamically based on binary replacement
target_col = 'ChurnLabel' if 'ChurnLabel' in df_final.columns else 'ChurnLabel_1'
X = df_final.drop(target_col, axis=1)
y = df_final[target_col]

# 7. Train/Test Split & Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. Evaluation
y_pred = model.predict(X_test)
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.2%}\n")
print(classification_report(y_test, y_pred))
