import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
import streamlit as st

st.title("Impulse Response Function (IRF) Visualizer")

# User input for two tickers
ticker1 = st.text_input("Enter first ticker (e.g., ^GSPC):", "^GSPC")
ticker2 = st.text_input("Enter second ticker (e.g., BTC-USD):", "BTC-USD")
period = st.text_input("Enter historical period (e.g., 1y, 2y, 5y):", "1y")

if st.button("Generate IRF"):
    try:
        # --- Fetch historical prices ---
        data = yf.Tickers(f"{ticker1} {ticker2}")
        hist = data.history(period=period)["Close"]

        if hist.empty:
            st.error("No data returned — check the tickers or period.")
        else:
            # --- Compute log returns ---
            log_return = np.log(hist / hist.shift(1)).dropna()

            # --- Fit VAR model ---
            model = VAR(log_return[[ticker1, ticker2]])
            results = model.fit(maxlags=5)
            irf = results.irf(10)  # 10-step horizon

            # --- Plot IRF ---
            fig = irf.plot(orth=True)  # no ax argument
            fig.set_size_inches(10, 6)
            st.pyplot(fig)

            st.success("Impulse Response plotted successfully!")

    except Exception as e:
        st.error(f"Error fetching or processing data: {e}")
