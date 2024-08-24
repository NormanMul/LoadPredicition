import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def load_data(file_path):
    print("Loading data...")
    data = pd.read_csv(file_path)
    print("Data loaded successfully.")
    return data

def preprocess_data(data):
    print("Preprocessing data...")
    # Assuming features have been selected and preprocessed appropriately
    features = ['nPaidOff', 'payFrequency_W', 'loanStatus_Paid Off Loan', 'clearfraudscore', 'loanAmount', 'loanAmount_PaidOffInteraction', 'paymentAmount']
    X = data[features]
    y = data['isFunded']  # Replace with the correct target variable

    print("Data preprocessed successfully.")
    return X, y

def train_model(X, y):
    print("Training model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model trained successfully.")
    return model

def save_model(model, file_path):
    print("Saving model...")
    joblib.dump(model, file_path)
    print("Model saved successfully.")

if __name__ == "__main__":
    try:
        data = load_data('loan_data.csv')  # Make sure this file is in the root directory of the project
        X, y = preprocess_data(data)
        print("Features used for training:", X.columns.tolist())  # Print the features used for training
        model = train_model(X, y)
        save_model(model, 'model.joblib')
        print("All steps completed successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
