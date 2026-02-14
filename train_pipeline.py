import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime

# Paths
DATA_PATH = "/mnt/ml-data/datasets/processed.csv"
MODEL_DIR = "/mnt/ml-data/models/"
LOG_DIR = "/mnt/ml-data/logs/"

def ensure_directories():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    target_col = 'income'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    le = LabelEncoder()
    y = le.fit_transform(y)

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, preprocessor

def train_model(model, X_train, y_train, X_test, y_test, preprocessor):
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return clf, accuracy

def log_results(model_name, accuracy):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_file = os.path.join(LOG_DIR, f"training_log_{timestamp}.txt")

    with open(log_file, "a") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("-" * 40 + "\n")

    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Timestamp: {timestamp}")

def main():
    ensure_directories()

    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42)
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        clf, accuracy = train_model(
            model, X_train, y_train, X_test, y_test, preprocessor
        )
        results[name] = accuracy
        trained_models[name] = clf

    best_model_name = max(results, key=results.get)
    best_accuracy = results[best_model_name]
    best_model = trained_models[best_model_name]
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Save best model
    model_path = os.path.join(MODEL_DIR, f"best_model_{timestamp}.pkl")
    joblib.dump(best_model, model_path)

    # Log results
    log_results(best_model_name, best_accuracy)

if __name__ == "__main__":
    main()
