# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder, StandardScaler
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN_TOURISM"))
DATASET_PATH = "hf://datasets/CRR79/TourismPackage-Purchase-Prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# ----- 1. Basic Cleaning -----
df.drop_duplicates(inplace=True)

for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# ----- 2. Feature Engineering -----
df['FamilySize'] = df['NumberOfPersonVisiting'] + df['NumberOfChildrenVisiting']
df['IncomePerPerson'] = df['MonthlyIncome'] / (df['FamilySize'] + 1)

# ----- 3. Define Features & Target -----
X = df.drop(["CustomerID", "ProdTaken"], axis=1)
y = df["ProdTaken"]

# Identify categorical and numeric features
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# ----- 4. Column Transformer (Preprocessing) -----
le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols)
    ], remainder='passthrough')


# ----- 5. Train-Test Split -----
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)# Save artifacts

joblib.dump(preprocessor,"preprocessor.joblib")



files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv","preprocessor.joblib"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="CRR79/TourismPackage-Purchase-Prediction",
        repo_type="dataset",
    )
# Save artifacts
#joblib.dump(preprocessor,"preprocessor.joblib")

#api.upload_file(
#        path_or_fileobj="preprocessor.joblib",
#        path_in_repo="preprocessor.joblib",  # just the filename
#        repo_id="CRR79/TourismPackage-Purchase-Prediction",
#        repo_type="model",
#)
