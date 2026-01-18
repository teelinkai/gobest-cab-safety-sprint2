import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 1. Set the Experiment Name (This organizes your runs in the UI)
mlflow.set_experiment("Gobest_Cab_Safety_Prediction")

# 2. Generate Dummy Data (Just to test the connection)
X, y = make_classification(n_samples=100, n_features=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. Start a Run
print("Starting MLflow Run...")
with mlflow.start_run():
    # A. Log Parameters (Hyperparameters)
    C_param = 0.5
    mlflow.log_param("C", C_param)
    mlflow.log_param("solver", "lbfgs")
    
    # B. Train Model
    model = LogisticRegression(C=C_param)
    model.fit(X_train, y_train)
    
    # C. Log Metrics (Performance)
    score = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", score)
    print(f"Logged Accuracy: {score}")
    
    # D. Log the Model itself (Artifact)
    mlflow.sklearn.log_model(model, "baseline_model")

print("Run Complete! Check MLflow UI.")