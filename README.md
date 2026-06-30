Uncertainty-Calibrated LSTM (AAPL)

An LSTM that forecasts Apple (AAPL) next-day price as a prediction interval
with a confidence level, rather than a single number. Uncertainty comes from
Monte Carlo dropout, and the focus is calibration - checking whether the
model's stated confidence matches reality. Includes a Streamlit dashboard
showing the forecast, its confidence band, and a calibration plot.

Install

bash
pip install -r requirements.txt


Run the dashboard

py -m streamlit run src/dashboard.py

