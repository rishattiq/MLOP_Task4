from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

app = Flask(__name__)

def train_model():
    # Load the dataset
    data = pd.read_csv("onlinefoods.csv")

    # Separate features (X) and target variable (y)
    X = data.drop(columns=["Output", "Feedback"])
    y = data["Output"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define preprocessing steps for numerical and categorical features
    numeric_features = ["Age", "Family size"]
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_features = ["Gender", "Marital Status", "Occupation", "Educational Qualifications"]
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Define the logistic regression model
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', LogisticRegression())])

    # Train the model
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, "trained_model.pkl")

    return model

# Load the trained model
model = train_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data sent in the request
        data = request.get_json()

        # Convert JSON data to DataFrame
        df = pd.DataFrame(data, index=[0])

        # Make predictions using the loaded model
        prediction = model.predict(df)[0]

        # Return the prediction as JSON response
        return jsonify({'prediction': prediction}), 200

    except Exception as e:
        # Return error message if prediction fails
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
