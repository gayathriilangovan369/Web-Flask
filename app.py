from flask import Flask, render_template, send_file, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open("Models/xgboost_model.pkl", "rb"))

# Load scaler
scaler = pickle.load(open("Models/scaler.pkl", "rb"))

# each plan always has the same monthly fee in the dataset
PLAN_FEES = {"Basic": 8.99, "Standard": 13.99, "Premium": 17.99}

# monthly watch hours of every customer, used to rank a subscriber against the dataset
ALL_WATCH_HOURS = pd.read_csv("data/Netflix.csv")["watch_hours"].dropna()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dataset")
def dataset():
    df = pd.read_csv("data/Netflix.csv")

    # first 20 rows, numbers rounded, empty cells shown as "-"
    sample = df.head(20).round(2).fillna("-")

    return render_template(
        "dataset.html",
        columns=list(df.columns),
        rows=sample.to_dict("records"),
        total_rows=f"{len(df):,}",
        total_columns=len(df.columns),
        churn_rate=round(df["churned"].mean() * 100, 1),
        missing_values=f"{int(df.isna().sum().sum()):,}",
    )


@app.route("/download")
def download_csv():
    # sends the whole CSV file to the visitor's computer
    return send_file("data/Netflix.csv", as_attachment=True, download_name="Netflix_customer_churn.csv")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    # GET = just show the empty form
    if request.method == "GET":
        return render_template("predict.html")

    # POST = the form was submitted
    watch_hours = float(request.form["watch_hours"])
    last_login_days = float(request.form["last_login_days"])
    plan = request.form["subscription_type"]
    number_of_profiles = int(request.form["number_of_profiles"])
    avg_watch_time_per_day = float(request.form["avg_watch_time_per_day"])
    payment_method = request.form["payment_method"]

    monthly_fee = PLAN_FEES[plan]
    subscription_type_Premium = 1 if plan == "Premium" else 0
    payment_method_Crypto = 1 if payment_method == "Crypto" else 0

    # same 7 features, same order as the notebook
    data = [[watch_hours, last_login_days, monthly_fee, subscription_type_Premium,
             number_of_profiles, avg_watch_time_per_day, payment_method_Crypto]]

    data = scaler.transform(data)
    probability = round(model.predict_proba(data)[0][1] * 100, 1)

    # percentage of subscribers who watch less than this one
    rank = round((ALL_WATCH_HOURS < watch_hours).mean() * 100)

    return render_template(
        "predict.html",
        probability=probability,
        watch_hours=watch_hours,
        last_login_days=last_login_days,
        plan=plan,
        number_of_profiles=number_of_profiles,
        payment_method=payment_method,
        monthly_fee=monthly_fee,
        rank=rank,
    )


@app.route("/models")
def models():
    return render_template("models.html")


@app.route("/insights")
def insights():
    return render_template("insights.html")


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
