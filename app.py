from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load and prepare data
data = pd.read_csv("credit_card.csv")

# Mapping transaction types
transaction_type_map = {"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5}
data["type"] = data["type"].map(transaction_type_map)

# Add a new feature for the balance difference
data["balanceDifference"] = data["oldbalanceOrg"] - data["newbalanceOrig"]

# Prepare features and target
features = ["type", "amount", "oldbalanceOrg", "newbalanceOrig", "balanceDifference"]
x = data[features].values
y = data["isFraud"].values

# Split the data
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)

# Train the model
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)

# Evaluate the model
ypred = model.predict(xtest)
print(classification_report(ytest, ypred))
print(confusion_matrix(ytest, ypred))

# Initialize Flask app
app = Flask(__name__)

# Home route to display the form
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        type = int(request.form['type'])
        amount = float(request.form['amount'])
        oldbalanceOrg = float(request.form['oldbalanceOrg'])
        newbalanceOrig = float(request.form['newbalanceOrig'])
        balanceDifference = oldbalanceOrg - newbalanceOrig

        # Handle different transaction types
        if type == 1 or type == 4:  # CASH_OUT or TRANSFER
            if balanceDifference != amount:
                prediction = "Fraud"
            else:
                features = np.array([[type, amount, oldbalanceOrg, newbalanceOrig, balanceDifference]])
                result = model.predict(features)[0]
                prediction = "Fraud" if result == 1 else "No Fraud"
        elif type == 3:  # CASH_IN
            if newbalanceOrig -oldbalanceOrg != amount:
                prediction = "Fraud"
            else:
                features = np.array([[type, amount, oldbalanceOrg, newbalanceOrig, balanceDifference]])
                result = model.predict(features)[0]
                prediction = "Fraud" if result == 1 else "No Fraud"
        elif type == 2 or type == 5:  # PAYMENT or DEBIT
            if balanceDifference != amount:
                prediction = "Fraud"
            else:
                features = np.array([[type, amount, oldbalanceOrg, newbalanceOrig, balanceDifference]])
                result = model.predict(features)[0]
                prediction = "Fraud" if result == 1 else "No Fraud"

    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
