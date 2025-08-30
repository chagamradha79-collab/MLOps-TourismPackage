# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
from sklearn.metrics import roc_auc_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import hf_hub_download
import mlflow
from sklearn.compose import make_column_transformer

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLOps-TourismPackage-experiment")

api = HfApi()


Xtrain_path = "hf://datasets/CRR79/TourismPackage-Purchase-Prediction/Xtrain.csv"
Xtest_path = "hf://datasets/CRR79/TourismPackage-Purchase-Prediction/Xtest.csv"
ytrain_path = "hf://datasets/CRR79/TourismPackage-Purchase-Prediction/ytrain.csv"
ytest_path = "hf://datasets/CRR79/TourismPackage-Purchase-Prediction/ytest.csv"

X_train = pd.read_csv(Xtrain_path)
X_test = pd.read_csv(Xtest_path)
y_train = pd.read_csv(ytrain_path)
y_test = pd.read_csv(ytest_path)


cat_features = [
"TypeofContact", "Occupation", "Gender", "MaritalStatus",
"Designation", "ProductPitched"
]
numeric_features = [
"Age", "CityTier", "NumberOfPersonVisiting", "PreferredPropertyStar",
"NumberOfTrips", "Passport", "OwnCar", "NumberOfChildrenVisiting",
"MonthlyIncome", "PitchSatisfactionScore", "NumberOfFollowups",
"DurationOfPitch"
]

#preprocessor = ColumnTransformer(
#    transformers=[
#        ('num', StandardScaler(), num_cols)
#    ], remainder='passthrough')

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), cat_features)
)
# ----- 6. Define Models -----
xgb_model = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)



# ----- 7. Hyperparameter Grids -----
param_grid = {
'model__n_estimators': [100, 200],
'model__max_depth': [3, 6, 10],
'model__learning_rate': [0.01, 0.1, 0.2]
}


model_name = "XGBoost"

#----- 8. Training, Hyperparameter Tuning & MLflow Tracking -----
with mlflow.start_run(run_name=model_name):
    print(f"\nTraining {model_name}...")
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", xgb_model)])

    grid_search = GridSearchCV(pipe, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

    # Log each combination as a separate MLflow run
    with mlflow.start_run(nested=True):
        mlflow.log_params(param_set)
        mlflow.log_metric("mean_test_score", mean_score)
        mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"Best Params: {grid_search.best_params_}")
    print("Accuracy:", acc)
    print("ROC-AUC:", roc_auc)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Log metrics & params to MLflow
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("roc_auc", roc_auc)


    # Save the model locally
    model_path = "best_TourismPackage_Purchase_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "CRR79/TourismPackage-Purchase-Prediction"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
               path_or_fileobj="best_TourismPackage_Purchase_model_v1.joblib",
               path_in_repo="best_TourismPackage_Purchase_model_v1.joblib",
               repo_id=repo_id,
               repo_type=repo_type,
               )
