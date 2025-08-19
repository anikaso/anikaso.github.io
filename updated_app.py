import streamlit as st
from new_custom_backend_model import run_full_pipeline, load_and_predict_from_csv

st.set_page_config(layout="wide")
st.title("📈 AI Stock Market Predictor (Custom Backend)")

symbol = st.text_input("Enter Ticker Symbol (e.g., AAPL, TSLA, GLD)", "GLD")
timeframe = st.selectbox("Select Timeframe", ['1h', '4h', '1d', '1w'])

if st.button("Run Prediction"):
    with st.spinner(f"Fetching and predicting for {symbol.upper()} at {timeframe}..."):
        run_full_pipeline(symbol, timeframe)
        fig, predictions_df, metrics = load_and_predict_from_csv(symbol, timeframe)
        st.pyplot(fig)
        avg_price = predictions_df['actual_close'].mean()
        price_accuracy = 100 - (metrics['mae'] / avg_price * 100)
        direction_accuracy = metrics['dir_acc']
        st.write("Metrics")
        st.write(f"Price Accuracy: {price_accuracy:.2f}%")
        st.write(f"Direction Accuracy: {direction_accuracy:.2f}%")
