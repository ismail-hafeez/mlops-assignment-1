import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data(filepath):
    """Loads the dataset from a CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """
    Preprocesses the data:
    - Separates features (X) and target (y).
    - Encodes the target variable.
    - Defines column transformer for numeric and categorical features.
    - Splits into train and test sets.
    """
    # Define target and features
    target_col = 'income'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode target variable
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Create preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, preprocessor

def train_evaluate_model(model, X_train, y_train, X_test, y_test, preprocessor, model_name):
    """Trains a model using a pipeline and evaluates its accuracy."""
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', model)])
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    return accuracy

def main():
    # Load data
    df = load_data('processed.csv')
    
    # Preprocess data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    
    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }
    
    results = {}
    
    # Train and evaluate models
    for name, model in models.items():
        accuracy = train_evaluate_model(model, X_train, y_train, X_test, y_test, preprocessor, name)
        results[name] = accuracy
        
    # Select best model
    best_model_name = max(results, key=results.get)
    best_accuracy = results[best_model_name]
    
    print(f"\nBest Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    main()
