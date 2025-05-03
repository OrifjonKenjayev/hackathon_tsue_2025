import pandas as pd
import joblib

# Load the saved model
model = joblib.load('linear_regression_model.pkl')

# Load the test data
test_data = pd.read_csv('test_data2.csv')

# Define features
features = ['Income', 'Rating', 'Cards', 'Age', 'Education', 'Gender', 'Student', 'Married', 'Ethnicity', 'Balance']

# Function to predict limit for a given ID
def predict_limit_by_id(input_id):
    """
    Predict credit limit for a given ID using the saved model and test data.
    
    Parameters:
    input_id (int): ID of the record
    
    Returns:
    float: Predicted credit limit
    """
    if input_id not in test_data['ID'].values:
        return f"ID {input_id} not found in the test dataset."
    
    # Get data for the given ID
    input_data = test_data[test_data['ID'] == input_id][features]
    
    # Predict using the loaded model
    predicted_limit = model.predict(input_data)[0]
    
    return predicted_limit

# Example predictions
print("Example Predictions using Saved Model:")
for id in [11111, 210, 281]:
    if id in test_data['ID'].values:
        prediction = predict_limit_by_id(id)
        print(f"Predicted Limit for ID {id}: {prediction:.2f}")
    else:
      print("ID", id ,"is not found")