import keras.models
import pandas as pd
import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime
from GoogleNews import GoogleNews
import locale

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error
from keras import Sequential
from keras.layers import LSTM, Dense

import pandas_ta as ta
from ta.volatility import BollingerBands
from ta.volatility import KeltnerChannel

sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

# customising googlenews api
googlenews = GoogleNews()
googlenews = GoogleNews(lang='en', region='IN')
googlenews = GoogleNews(encode='utf-8')


# create custom symbol formatter
def format_ticker(symbol):
    return symbol + '.NS'


class StockApp:
    def __init__(self):
        self.apply_lstm = None
        self.get_news = None
        self.vis_plots = None
        self.Symbol = None
        self.Ticker = None
        self.info = None
        self.period = None
        self.display_info = False
        self.tickers = pd.read_csv('National_Stock_Exchange_of_India_Ltd.csv')['Symbol'].tolist()

    def run(self):
        st.title('NIFTY50 STOCK MARKET ANALYSIS', anchor=False, help='Analysis of India\'s top 50 stocks listed in NSE')
        # store the selected stock symbol
        self.Symbol = st.selectbox('Select your stock', options=tuple(self.tickers))
        # store the information on the company
        self.info = yf.Ticker(format_ticker(self.Symbol)).info
        # load the sidebar
        self.custom_sidebar()

        # get the stock data based on the selected timeframe
        end_date = datetime.now()
        start_date = datetime(end_date.year - self.period, end_date.month, end_date.day)
        fin_data = yf.download(format_ticker(self.Symbol), start_date, end_date)

        # plot the close price chart to show the trend of the stock in the market
        self.plot_close_price(fin_data)

        # fetch customized information on the company
        self.get_stock_info()

        # plot various plots for more deep analysis of the stock
        self.visualization_plots(fin_data)

        # fetch news about the company
        self.fetch_news()

        # apply the lstm model on the stock data and predict the close price
        if self.apply_lstm:
            self.train_model(fin_data)

    def custom_sidebar(self):
        sidebar = st.sidebar
        sidebar.subheader('Options')
        # range slider for selecting the period
        self.period = sidebar.slider(label='Select period:', min_value=1, max_value=10)
        self.display_info = sidebar.checkbox(f'Display Additional Information on {self.info["longName"]}', False)
        self.get_news = sidebar.checkbox('Read News')
        sidebar.subheader('Analytical Charts')
        self.vis_plots = sidebar.multiselect('Select:', ['Moving Average', 'Average Daily Returns', 'Volume Sales',
                                                         'Adjusted Closing Price', 'Technical Indicators'])
        self.apply_lstm = sidebar.checkbox('Apply LSTM', False)

    def get_stock_info(self):
        st.write("***")
        st.subheader('Performance')
        st.write("")
        info_cnt = st.container(height=150, border=False)
        col1, col2 = info_cnt.columns(2, gap='large', vertical_alignment='top')
        # display company performance indicators
        with col1:
            st.write('Current Price: ', self.info['currentPrice'])
            st.write('Total Revenue : ', self.info['totalRevenue'])
            st.write('Debt To Equity : ', self.info['debtToEquity'])
            st.write('Earnings Growth: ', self.info['earningsGrowth'])
        with col2:
            st.write('52 Week High: ', self.info['fiftyTwoWeekHigh'])
            st.write('52 Week Low: ', self.info['fiftyTwoWeekLow'])
            st.write('Market Cap. : ', self.info['marketCap'])
            st.write('Volume : ', self.info['volume'])
        if self.display_info:
            st.write("***")
            st.subheader('Company Description')
            st.write("")
            st.write(self.info['longBusinessSummary'])
            st.write("")
            st.write(f"Company website: {self.info['website']}")

    # plot graph for close price
    def plot_close_price(self, df):
        fig = make_subplots(rows=1, cols=1, print_grid=True)
        fig.update_layout(title=f"{self.info['longName']}")
        fig.add_trace(go.Line(x=df.index, y=df['Close'], line=dict(color='green')))
        fig.update_yaxes(range=[0, 1000000000], secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    def fetch_news(self):
        if self.get_news:
            st.write("***")
            st.subheader("News")
            st.write("")
            googlenews.search(self.info['longName'])
            results = googlenews.results()
            for news in results:
                st.markdown(f"**{news['title']}**")
                st.write(f"{news['desc']}")
                st.write(f"Link: {news['link']}")
                st.write("***")

    # Visualising technical indicators
    def strategy_plots(self):
        # take the past year data for the stock
        end_date = datetime.now()
        start_date = datetime(end_date.year - 1, end_date.month, end_date.day)
        df = yf.download(format_ticker(self.Symbol), start_date, end_date)

        # MACD strategy
        df.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True)
        # RSI strategy
        df['RSI'] = ta.rsi(df['Close'], length=14)
        # Bollinger Bands Strategy
        indicator_bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['BB_mid'] = indicator_bb.bollinger_mavg()
        df['BB_high'] = indicator_bb.bollinger_hband()
        df['BB_low'] = indicator_bb.bollinger_lband()
        # Keltner Strategy
        indicator_keltner = KeltnerChannel(high=df['High'], low=df['Low'], close=df['Close'], window=20)
        df['Keltner_mid'] = indicator_keltner.keltner_channel_mband()
        df['Keltner_high'] = indicator_keltner.keltner_channel_hband()
        df['Keltner_low'] = indicator_keltner.keltner_channel_lband()

        fig = make_subplots(rows=5, cols=1,
                            subplot_titles=['Close', 'MACD', 'RSI', 'Bollinger Bands', 'Keltner Channels', ])
        #  Plot close price
        fig.add_trace(go.Line(x=df.index, y=df['Close'], line=dict(color="blue", width=1), name="Close"), row=1, col=1)
        # Plot MACD
        fig.add_trace(go.Line(x=df.index, y=df['MACD_12_26_9'], line=dict(color="#99b3ff", width=1), name="MACD"),
                      row=2, col=1)
        fig.add_trace(go.Line(x=df.index, y=df['MACDs_12_26_9'], line=dict(color="#ebab34", width=1), name="MACD"),
                      row=2, col=1)
        # Plot RSI
        fig.add_trace(go.Line(x=df.index, y=df['RSI'], line=dict(color="#99b3ff", width=1), name="RSI"), row=3, col=1)
        fig.add_hline(y=30, line=dict(color="#ebab34", width=1), row=3, col=1)
        fig.add_hline(y=70, line=dict(color="#ebab34", width=1), row=3, col=1)
        # Plot Bollinger
        fig.add_trace(go.Line(x=df.index, y=df['Close'], line=dict(color="blue", width=1), name="Close"), row=4, col=1)
        fig.add_trace(go.Line(x=df.index, y=df['BB_high'], line=dict(color="#ebab34", width=1), name="BB High"), row=4,
                      col=1)
        fig.add_trace(go.Line(x=df.index, y=df['BB_mid'], line=dict(color="#fac655", width=1), name="BB Mid"), row=4,
                      col=1)
        fig.add_trace(go.Line(x=df.index, y=df['BB_low'], line=dict(color="#ebab34", width=1), name="BB Low"), row=4,
                      col=1)
        # Plot Keltner
        fig.add_trace(go.Line(x=df.index, y=df['Close'], line=dict(color="blue", width=1), name="Close"), row=5, col=1)
        fig.add_trace(
            go.Line(x=df.index, y=df['Keltner_high'], line=dict(color="#ebab34", width=1), name="Keltner High"),
            row=5, col=1)
        fig.add_trace(go.Line(x=df.index, y=df['Keltner_mid'], line=dict(color="#fac655", width=1), name="Keltner Mid"),
                      row=5, col=1)
        fig.add_trace(go.Line(x=df.index, y=df['Keltner_low'], line=dict(color="#ebab34", width=1), name="Keltner Low"),
                      row=5, col=1)
        fig.update_layout(
            title={'text': self.Symbol},
            autosize=False, width=800, height=1600)
        fig.update_yaxes(range=[0, 1000000000], secondary_y=True)
        fig.update_yaxes(visible=False, secondary_y=True)  # hide range slider
        st.plotly_chart(fig, use_container_width=True)

    def visualization_plots(self, df: pd.DataFrame):
        stock = df.copy()
        if self.vis_plots:
            st.write("***")
            st.subheader('Visualisation Charts')
            st.write("")
        # plot adjusted close price graph
        if "Adjusted Closing Price" in self.vis_plots:
            st.subheader(f'Historical Data On Adjusted Closing Price of {self.info["longName"]}', anchor=False)
            fig = plt.figure(figsize=(15, 6))
            stock['Adj Close'].plot()
            plt.title(f'{self.info["longName"]} stock Adjusted Closing Price ')
            plt.xlabel(None)
            plt.tight_layout()
            st.pyplot(fig)
            st.write("")
        # plot moving average graph
        if "Moving Average" in self.vis_plots:
            st.subheader(f"Moving Average Data of {self.info['longName']}", anchor=False)
            ma_days = [10, 30, 50]
            for ma in ma_days:
                stock[f'SMA{ma}'] = stock['Adj Close'].rolling(ma).mean()
            fig = plt.figure(figsize=(15, 6))
            stock['Adj Close'].plot(legend=True)
            stock['SMA10'].plot(legend=True)
            stock['SMA30'].plot(legend=True)
            stock['SMA50'].plot(legend=True)
            plt.title(f'{self.info["longName"]} Moving Average')
            st.pyplot(fig)
            st.write("")

        # plot average daily returns graph
        if "Average Daily Returns" in self.vis_plots:
            st.subheader(f"Average Daily Returns Data of {self.info['longName']}", anchor=False)
            fig = plt.figure(figsize=(15, 6))
            stock['Adj Close'].pct_change().plot()
            plt.title(f'{self.info["longName"]} Average Daily Returns')
            st.pyplot(fig)
            st.write("")

        # plot volume sales graph
        if "Volume Sales" in self.vis_plots:
            st.subheader(f"Volume Sales of {self.info['longName']}", anchor=False)
            fig = plt.figure(figsize=(15, 6))
            stock['Volume'].plot()
            plt.title(f'{self.info["longName"]} Volume Sales')
            st.pyplot(fig)
            st.write("")
        # plot graphs of the technical indicators
        if "Technical Indicators" in self.vis_plots:
            st.subheader(f"Technical Indicators", anchor=False)
            self.strategy_plots()

    # Model Training
    @staticmethod
    def train_model(df: pd.DataFrame):
        # loading the data
        data = pd.DataFrame(df['Close'])
        # Cleaning the dataset
        data.fillna(0, inplace=True)
        close = data.values

        # scaling the data to a range between 0 to 1
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(close)

        # splitting the dataset for model training
        train_size = int(0.95 * len(scaled_data))
        train_data = scaled_data[:train_size, :]
        test_data = scaled_data[train_size - 60:, :]

        x_train, x_test = [], []
        y_train, y_test = [], []

        for i in range(60, len(train_data)):
            x_train.append(train_data[i - 60:i, 0])
            y_train.append(train_data[i, 0])

        y_test = close[train_size:, :]
        for i in range(60, len(test_data)):
            x_test.append(test_data[i - 60:i, 0])

        # converting the train and test data into numpy arrays
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        x_test, y_test = np.array(x_test), np.array(y_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # Creating LSTM model
        model = Sequential()
        model.add(LSTM(128, input_shape=(x_train.shape[1], 1), return_sequences=True))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        # compiling the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # fitting the data to the model and saving the fitted model
        history = model.fit(x_train, y_train, batch_size=1, epochs=10, verbose=1)
        # model.save('lstm_model.h5')

        # predicting close price using test data
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        # evaluating the model
        print('RMSE: ', root_mean_squared_error(y_test, predictions))

        # creating result data
        train = data[:train_size]
        valid = data[train_size:]
        valid['Predictions'] = predictions

        # plotting the predicted close price graph
        st.write("***")
        st.subheader("LSTM Forecasting")
        st.write("")
        fig = plt.figure(figsize=(16, 6))
        plt.title('Closing Price Prediction Using LSTM model')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price in INR', fontsize=18)
        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        plt.tight_layout()
        st.pyplot(fig)


if __name__ == '__main__':
    app = StockApp()
    app.run()
