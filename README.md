# Stock Price Prediction App using LSTM

This project involves building a stock price prediction web application using Long Short-Term Memory (LSTM) neural networks. The app provides users with insights into historical stock data, visualizations of closing prices over time, moving averages, and predictions of future stock prices based on the LSTM model.

## Features

1. **Data Description:** The app offers a summary of stock data from a specified start date to an end date, giving users a comprehensive understanding of the historical trends.
2. **Visualizations:**
   - Closing Price vs Time Chart: Users can visualize the closing prices of the stock over time, helping them identify trends and patterns.
   - Closing Price vs Time Chart with Moving Averages: The app displays the closing prices along with the 100-day and 200-day moving averages, providing additional insights into the stock's performance.
3. **Prediction:**
   - Future stock prices are predicted based on historical data using an LSTM model, enabling users to anticipate potential market movements.
   - The app also compares predicted prices with actual prices to evaluate the model's performance, allowing users to gauge the reliability of the predictions.

## Data Source

I obtained the stock price data from Yahoo Finance using the `yfinance` library. Users can input the stock ticker symbol to fetch data for analysis and prediction.

## Approaches

1. Fetched historical stock price data for the specified stock ticker symbol and preprocessed it for training and testing the LSTM model, ensuring the data quality and consistency.
2. Built and trained an LSTM model using historical stock price data, leveraging the power of deep learning to capture complex patterns and relationships in the data.
3. Evaluated the trained model's performance using testing data and visualized the predictions compared to actual stock prices, providing users with valuable insights into the model's accuracy and reliability.
4. The app is built using the Streamlit framework.
   
## Note

- Install the required libraries (`pandas`, `matplotlib`, `yfinance`, `keras`, `streamlit`) using `pip`.

## Conclusion

In conclusion, my stock price prediction app provides a user-friendly interface for investors and traders to analyze historical stock data and make informed decisions. By leveraging LSTM neural networks, the app offers accurate predictions of future stock prices, aiding users in their investment strategies. I'm proud of the insights and functionalities the app offers, and I'm excited to continue refining and improving it in the future.
