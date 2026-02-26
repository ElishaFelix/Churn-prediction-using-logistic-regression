
from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load data
data = pd.read_csv("data/customer_churn_data.csv")

X = data.drop("churn", axis=1)
y = data["churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

@app.route("/")
def home():
    return render_template("index.html", accuracy=round(accuracy, 2))

@app.route("/predict", methods=["POST"])
def predict():
    tenure = float(request.form["tenure"])
    monthly = float(request.form["monthly"])
    logins = float(request.form["logins"])
    tickets = float(request.form["tickets"])

    input_data = [[tenure, monthly, logins, tickets]]
    prediction = model.predict(input_data)[0]

    result = "Customer is likely to churn." if prediction == 1 else "Customer is likely to stay."
    return render_template("index.html", prediction=result, accuracy=round(accuracy, 2))

if __name__ == "__main__":
    app.run(debug=True)
