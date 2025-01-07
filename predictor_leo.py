from flask import Flask, request, render_template
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pickle import load

app = Flask(__name__)
#model = load(open("../models/random_forest_regressor_42.sav", "rb"))


@app.route("/", methods = ["GET", "POST"])
def index():
    if request.method == "POST":
        val1 = int(request.form["STEP"])
        ticker = "MXN=X"
        data = yf.Ticker(ticker)
        historical_data = data.history(period="10y")
        historical_data = historical_data[["Close"]]
        historical_data.rename(columns={"Close": "MXN_USD"}, inplace=True)
        models = ExponentialSmoothing(historical_data['MXN_USD'],seasonal_periods=12 ,trend='add', seasonal='add').fit()
        prediction = models.forecast(val1)
        prediction = prediction.values

    else:
        prediction = 0
    return render_template("front.html", prediction = prediction)