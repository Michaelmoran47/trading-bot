import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import random

class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'  # Resets color

def load_and_prepare_data(filename='btc_features.csv'):
    """
    Load the feature data and prepare for training
    """
    print("Loading data...")
    df = pd.read_csv(filename)
    
    # Drop timestamp (not a feature)
    df = df.drop('timestamp', axis=1)
    
    # Separate features (X) and target (y)
    X = df.drop('target', axis=1)
    y = df['target']
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"\nTarget distribution:")
    print(y.value_counts())
    print(f"Baseline accuracy: {y.value_counts().max() / len(y):.4f}")
    
    return X, y

def train_model(X_train, y_train):
    """
    Train a Random Forest classifier
    """
    print("\nTraining Random Forest model...")
    
    # Initialize the model
    model = RandomForestClassifier(
        n_estimators=100,      # Number of trees
        max_depth=10,          # Maximum depth of trees
        min_samples_split=20,  # Minimum samples to split a node
        random_state=random.randint(0, 1000000000),       # For reproducibility
        n_jobs=-1              # Use all CPU cores
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    print("Training complete!")
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    if accuracy > 0.55:
        color = Colors.GREEN
    elif accuracy > 0.52:
        color = Colors.YELLOW
    else:
        color = Colors.RED
    
    print(f"\n{Colors.BOLD}Accuracy: {color}{accuracy:.4f}{Colors.END}")

    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Down (0)', 'Up (1)']))
    
    # Confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("\nInterpretation:")
    print(f"  True Negatives (predicted down, was down): {cm[0,0]}")
    print(f"  False Positives (predicted up, was down): {cm[0,1]}")
    print(f"  False Negatives (predicted down, was up): {cm[1,0]}")
    print(f"  True Positives (predicted up, was up): {cm[1,1]}")
    
    return accuracy

def feature_importance(model, feature_names):
    """
    Show which features are most important
    """
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE")
    print("="*50)
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create dataframe and sort
    feature_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 most important features:")
    print(feature_imp.head(10).to_string(index=False))
    
    return feature_imp

def save_model(model, filename='trained_model.pkl'):
    """
    Save the trained model to disk
    """
    joblib.dump(model, filename)
    print(f"\nModel saved to {filename}")

if __name__ == "__main__":
    # Load data
    X, y = load_and_prepare_data()
    
    # Split data chronologically (important for time series!)
    # Train on first 80%, test on last 20%
    split_index = int(len(X) * 0.8)
    
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate on test set
    accuracy = evaluate_model(model, X_test, y_test)
    
    # Show feature importance
    feature_imp = feature_importance(model, X.columns)
    
    # Save the model
    save_model(model)
    
    print("\n" + "="*50)
    print("DONE!")
    print("="*50)