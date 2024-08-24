# Loan Quality Prediction App

This is a Streamlit app to predict the quality/outcome of loans based on input features.

## Files

- `app.py`: Streamlit app to upload data and get predictions.
- `demo.py`: Script to train and save the machine learning model.
- `requirements.txt`: List of dependencies.
- `README.md`: Description of the project.
- `setup.sh`: Shell script for setting up the environment on Streamlit Cloud.

## Setup

1. Clone the repository.
2. Install the dependencies: `pip install -r requirements.txt`.
3. Train the model by running `demo.py` (Ensure the dataset is named `loan_data.csv`): `python demo.py`.
4. Run the Streamlit app: `streamlit run app.py`.

## Author

Customized for loan prediction using a RandomForest model.
