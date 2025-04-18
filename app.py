from flask import Flask, render_template, request
from utils.data_loader import get_stock_list
from model.lstm_model import predict_stock_price

app = Flask(__name__)
stock_list = get_stock_list()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    selected_ticker = None
    plot1 = plot2 = plot3 = None

    if request.method == 'POST':
        selected_ticker = request.form['ticker']
        prediction, plot1, plot2, plot3 = predict_stock_price(selected_ticker)

    return render_template(
        'index.html',
        prediction=prediction,
        stocks=stock_list,
        selected=selected_ticker,
        plot1=plot1,
        plot2=plot2,
        plot3=plot3
    )

if __name__ == '__main__':
    app.run(debug=True)
