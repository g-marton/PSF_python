import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import pandas_ta as ta
from prophet import Prophet
from datetime import date, datetime, timedelta
from prophet.plot import plot_plotly
from plotly import graph_objects as go
import quantstats as qs
import matplotlib.pyplot as plt
import seaborn as sns
from stocknews import StockNews
import requests
from pypfopt import expected_returns
from pypfopt import risk_models
from pypfopt import EfficientFrontier, objective_functions
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import plotting
import scipy.stats as stats
from pypfopt import black_litterman, risk_models
from pypfopt import BlackLittermanModel
import matplotlib as mpl
import matplotlib.font_manager as fm
import os

os.system('apt-get update && apt-get install -y ttf-mscorefonts-installer fontconfig && fc-cache -f -v')

mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.unicode_minus': False
})

plt.rcParams['font.family'] = 'DejaVu Sans'

qs.extend_pandas()

st.write(fm.findSystemFonts(fontpaths=None, fontext='ttf'))

#i have to figure out why the shrinkage makes the correlation matrix so negative and how can I make it better

#what i think I have to do still: I still have to do something with the dividends in option 1, and then I still have to do something with option 3 i think, for the portfolio performance analysis, as of to define when we invested into each individual stock

#dates to be fixed at the top
#indicators to show both at the same time

#page configuration and opening 
st.set_page_config(page_title="PSF DASHBOARD", page_icon=":smile:", layout="wide")
st.title("Welcome to Prosper Social Finance's Interactive Financial Analysis Dashboard!")
col1, col2 = st.columns([0.14, 0.86], gap="small")
col1.write("code done by:")
linkedin = "https://www.linkedin.com/in/gergely-marton-2024-2026edi"
col2.markdown(
        f'<a href="{linkedin}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="15" height="15" style="vertical-align: middle; margin-right: 10px;">`Gergely Marton`</a>',
        unsafe_allow_html=True,
    )
selection = st.selectbox("Please select the analysis tools you would like to use: ", ["Balloons", "Opt1: Basic technical single stock analysis", "Opt2: Stock Price Forecasting with Prophet (developed by Meta)", "Opt3: Advanced technical single stock analysis", "Opt4: Advanced portfolio analysis", "Opt5: Simple Efficient Frontier Portfolio Optimization", "Opt6: Black-Litterman Efficient Frontier Portfolio Optimization"])

############################################################################################################################################
if selection == "Balloons":
    st.balloons()
############################################################################################################################################

############################################################################################################################################
#basic analysis section to see
if selection == "Opt1: Basic technical single stock analysis":
    
    st.title("Main section of the dashboard:")

    ticker = st.sidebar.text_input('Please input the ticker symbol for the stock (as found on yahoofinance):', value='MSFT')
    start_date = st.sidebar.date_input('Start date of close price retrievals:',value="2020-01-01")
    end_date = st.sidebar.date_input('End date of close price retrievals:')

    data = yf.download(ticker, start = start_date, end = end_date)
    fig = px.line(data, x = data.index, y = data['Close'].squeeze(), title = ticker, labels={'y': 'Close'})
    
    st.subheader("Daily stock close price graph:")
    st.plotly_chart(fig)

    pricing_data, technical, news, other = st.tabs(['Pricing Data', 'Technical Analysis Dashboard', 'News that might be relevant', 'Statements and other data'])

    with pricing_data: 
        st.subheader('Price Data Dashboard:')
        st.write('Price data of the stock for the specified period:')
        data.columns = ['_'.join(col).strip() for col in data.columns]
        print(data.columns)

        data2 = data.copy()
        data2['% Log Returns'] = np.log(data[f"Close_{ticker}"] / data[f"Close_{ticker}"].shift(1))
        data2.dropna(inplace = True)
        st.write(data2)

        st.subheader("Basic calculation results:")
        annual_return = data2['% Log Returns'].mean()*252*100
        st.write("Annual Continuously Compounded Return is:", annual_return, '%')
        stdev = np.std(data2['% Log Returns'])*np.sqrt(252)
        st.write('The Annual Standard Deviation of the stock is:', stdev*100, '%')
        st.write('The Risk Adjusted Return is:', annual_return/(stdev*100))

    with technical: 
        st.subheader('Technical Analysis Dashboard:')
        df= pd.DataFrame()
        indicator_list=df.ta.indicators(as_list=True)
        excluded_indicators = ["aberration", "above", "above_value", "accbands", "adjusted_sortino", "adx", "aggregate_returns", "amat", "aobv", "apo", "aroon", "avg_loss", "avg_return", "avg_win", "bbands", "below", "below_value", "best", "bop", "brar", "cagr", "calmar", "cdl_pattern", "cdl_z", "cksp", "common_sense_ratio", "comp", "compare", "compsum", "conditional_value_at_risk", "consecutive_losses", "consecutive_wins", "cpc_index", "cross", "cross_value", "curr_month", "cvar", "date", "dm", "donchian", "eri", "expected_return", "expected_shortfall", "exponential_stdev", "exposure", "fisher", "gain_to_pain_ratio", "geometric_mean", "ghpr", "greeks", "ha", "hilo", "hwc", "ichimoku", "implied_volatility", "information_ratio", "kc", "kdj", "kelly_criterion", "kst", "kvo", "log_returns", "long_run", "macd", "massi", "max_drawdown", "mcgd", "metrics", "monthly_returns", "mtd", "multi_shift", "ohlc4", "omega", "outlier_loss_ratio", "outlier_win_ratio", "outliers", "payoff_ratio", "pct_rank", "pdist", "plot_daily_returns", "plot_distribution", "plot_drawdown", "plot_drawdown_periods", "plot_earnings", "plot_histogram", "plot_log_returns", "plot_monthly_heatmap", "plot_returns", "plot_rolling_beta", "plot_rolling_sharpe", "plot_rolling_sortino", "plot_rolling_volatility", "plot_snapshot", "plot_yearly_returns", "ppo", "probabilistic_adjusted_sortino_ratio", "probabilistic_sharpe_ratio","probabilistic_sortino_ratio", "profit_factor", "profit_ratio", "psar", "pvo", "pvr", "qqe", "qstick", "qtd", "r2", "r_squared", "rar", "rebase", "recovery_factor", "remove_outliers", "risk_of_ruin", "risk_return_ratio", "rolling_greeks", "rolling_sharpe", "rolling_sortino", "rolling_volatility", "ror", "rvgi", "serenity_index", "sharpe", "short_run", "smart_sharpe", "smart_sortino", "smi", "sortino", "squeeze", "squeeze_pro", "stc", "stoch","stochrsi", "supertrend", "tail_ratio", "td_seq", "thermo", "to_drawdown_series", "to_excess_returns","to_log_returns", "to_prices", "to_returns", "tos_stdevall", "treynor_ratio", "trix", "tsi", "tsignals", "ulcer_index", "ulcer_performance_index", "upi", "value_at_risk", "var", "vortex", "vp", "win_loss_ratio", "win_rate", "worst", "xsignals", "ytd"]
        filtered_indicator_list = [ind for ind in indicator_list if ind not in excluded_indicators]
        selected_indicators = st.multiselect('Please select the technical indicator you are interested in:', options=filtered_indicator_list)
        indicators_df = pd.DataFrame(index=data.index)
        
        for indicator_name in selected_indicators:
            method = getattr(ta, indicator_name)
            indicator = method(
                low=data[f"Low_{ticker}"], 
                close=data[f"Close_{ticker}"], 
                high=data[f"High_{ticker}"], 
                open=data[f"Open_{ticker}"], 
                volume=data[f"Volume_{ticker}"]
            )
        
            # Add the indicator to the DataFrame
            indicators_df[indicator_name] = indicator
    
        # Combine indicators with historical price data
        indicators_df["Close"] = data[f"Close_{ticker}"]

        # Plot all data together
        fig = px.line(indicators_df, title=f"Technical Indicators for {ticker}")
        for col in indicators_df.columns:
            fig.add_scatter(x=indicators_df.index, y=indicators_df[col], name=col)

        # Show the plot
        st.plotly_chart(fig)

        # Display the calculated indicator values in a table
        st.subheader("Indicator values in a table for the specified period:")
        st.write(indicators_df)

    with news:
        sn = StockNews(ticker, save_news=False)
        df_news = sn.read_rss()
        for i in range(10):
            st.subheader(f'News {i+1}')
            st.write(df_news['published'][i])
            st.write(df_news['title'][i])
            st.write(df_news['summary'][i])
            title_sentiment = df_news['sentiment_title'][i]
            st.write(f'Title sentiment: {title_sentiment}')
            news_sentiment = df_news['sentiment_summary'][i]
            st.write(f'News sentiment: {news_sentiment}')
    
    with other:
        base_url = 'https://financialmodelingprep.com/api'

        st.header('Financial Modeling Stock Screener')
        financial_data = st.sidebar.selectbox('Financial Data Type:', options = ('income-statement', 'balance-sheet-statement', 
                                                                                 'cash-flow-statement', 'historical-price-full/stock_dividend', 
                                                                                 'income-statement-growth', 'balance-sheet-statement-growth', 'cash-flow-statement-growth'
                                                                                 'key-metrics', 'key-metrics-ttm', 'ratios-ttm', 'ratios', 'financial-growth', 
                                                                                 'quote', 'enterprise-values', 'rating', 'historical-rating', 'discounted-cash-flow', 
                                                                                 'historical-discounted-cash-flow-statement', 'historical-price-full', 'historical price smaller intervals'))

        if financial_data == 'historical price smaller intervals': 
            interval = st.sidebar.selectbox('Interval', options=('1min', '5min', '15min', '30min', '1hour', '4hour'))
            financial_data = 'historical-chart/'+interval

        API_KEY = 'vSYSW0hsqQXc5N32P4jMXXuv0pOqPwXe'
        url = f'{base_url}/v3/{financial_data}/{ticker}?apikey={API_KEY}'
        response = requests.get(url)
        data = response.json()
        df = pd.DataFrame(data).T

        if financial_data != 'historical-price-full/stock_dividend':
            st.write(df)

        elif financial_data == 'historical-price-full/stock_dividend':
            st.header("Dividend calculator for the stock")
            start_date_dividend = st.date_input("Please select the starting date from which we should add up dividends:")
            historical_dividends = data.get('historical', [])

            if not historical_dividends:
                st.write('No dividend data available for the selected stock.')
            else:
                # Convert to DataFrame
                df_2 = pd.DataFrame(historical_dividends)

                # Filter by Start Date
                df_2['date'] = pd.to_datetime(df_2['date'])
                filtered_dividends = df_2[df_2['date'] >= pd.to_datetime(start_date_dividend)]

                # Sum Dividends
                total_dividends = filtered_dividends['adjDividend'].sum()

                # Display Results
                st.write(f'Total Dividends Received for a single stock of {ticker} since {start_date_dividend}: ${total_dividends:.2f}')
                st.write('Dividend Details:')
                st.write(filtered_dividends[['date', 'adjDividend', 'dividend']])
############################################################################################################################################

############################################################################################################################################
if selection == "Opt2: Stock Price Forecasting with Prophet (developed by Meta)":
    
    st.title("Main section of the dashboard:")

    st.header("Stock Price Forecasting using Prophet")

    st.write("The stock price prediction method using the Prophet library in Python involves forecasting future prices by modeling the historical time series data of stock prices. Prophet decomposes the time series into three main components: trend (long-term growth or decline), seasonality (recurring patterns such as daily or yearly cycles), and holiday effects (irregular deviations due to events).")

    st.write("It fits a statistical model to this data and predicts future values by extrapolating the trend and seasonality while accounting for uncertainty. Although Prophet is robust, intuitive, and handles missing data and outliers well, it is primarily a statistical forecasting method and may struggle with highly volatile or news-driven stock price data.")

    st.write("The Prophet library, developed by Facebook (now Meta), is primarily a time series forecasting tool. \n Prophet can give reasonable forecasts for stocks with clear trends or seasonality, but its predictive power is limited for highly random or news-driven markets. \n It fits a predefined additive structure to your time series and extrapolates it forward, making it robust, flexible, and interpretable.")
    
    st.subheader("Inputs:")
    start = st.date_input("Please provide the starting date for stock data retrieval:", value="2020-01-01")
    today = date.today().strftime("%Y-%m-%d")
    stocks = st.text_input("Please input a single ticker symbol for the stock you are interested in (as found on yahoofinance):", value="MSFT")
    n_years = st.slider("Please select the years for prediction:", 1, 4)
    period = n_years*365

    #@st.cache_data
    def load_data(ticker):
        data = yf.download(ticker, start, today)
        data.reset_index(inplace=True)
        return data

    data_load_state = st.text("Load data...")
    data = load_data(stocks)
    data_load_state.text("Loading data...done!")

    st.subheader("Raw Data last 5")
    data.columns = ['_'.join(col).strip() for col in data.columns]
    st.write(data.tail())

    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date_'], y=data[f'Open_{stocks}'], name='stock_open'))
        fig.add_trace(go.Scatter(x=data['Date_'], y=data[f'Close_{stocks}'], name='stock_close'))
        fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()

    #Forecasting

    df_train = data.loc[:, ["Date_", f"Close_{stocks}"]].copy()
    df_train = df_train.rename(columns={"Date_": "ds", f"Close_{stocks}": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    st.subheader('Forecast Data last 5')
    st.write(forecast.tail())

    st.subheader('Predicted movement depicted on a graph')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.subheader('Forecast components')
    fig2 = m.plot_components(forecast)
    st.write(fig2)
############################################################################################################################################

############################################################################################################################################
if selection == "Opt3: Advanced technical single stock analysis":

    st.title("Main section of the dashboard:")

    st.header("Advanced Stock Analysis")

    st.write("Here we can select specific graphs and plots to receive insightful data for a specific stock.")

    ticker = st.sidebar.text_input("Ticker (blue line on graphs):", value='MSFT')
    benchmark_ticker = st.sidebar.text_input("Benchmark Ticker (e.g., SPY) (yellow line on graphs):", value="SPY")
    period = st.sidebar.selectbox("Select Period:", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"])
    confidence_interval = st.sidebar.selectbox("Select confidence interval for expected shortfall and VaR:", [0.9, 0.95, 0.99])
    frequency = st.sidebar.selectbox("Select for the frequency for VaR and expected shortfall in trading days:", [1, 21, 63, 126, 252])

    stock = qs.utils.download_returns(ticker, period = period)
    stock2 = qs.utils.download_returns(ticker, period= "5y")
    benchmark = qs.utils.download_returns(benchmark_ticker, period=period)

    st.subheader(f"Key Metrics for {ticker}:")
    sharpe = qs.stats.sharpe(stock)
    calmar = qs.stats.calmar(stock)
    cagr = qs.stats.cagr(stock)
    volatility = qs.stats.volatility(stock)
    kurtosis = qs.stats.kurtosis(stock)
    skewness = qs.stats.skew(stock)
    av_return = qs.stats.avg_return(stock)
    av_loss = qs.stats.avg_loss(stock)
    av_win = qs.stats.avg_win(stock)
    win_loss_ratio = qs.stats.win_loss_ratio(stock)
    es = qs.stats.expected_shortfall(stock2, confidence=confidence_interval)
    es_adjusted = es * (frequency ** 0.5)
    VaR = qs.stats.value_at_risk(stock2, confidence=confidence_interval)
    VaR_adjusted = VaR * (frequency ** 0.5)
    max_drawdown = qs.stats.max_drawdown(stock)


    st.write(f"Sharpe ratio for the past {period} period:", sharpe)
    st.write(f"Calmar ratio for the past {period} period:", calmar)
    st.write(f"The compound annual growth rate for the past {period} period: {cagr*100:.2f}%")
    st.write(f"The annualized volatility of the stock based on the past {period} period: {volatility*100:.2f}%")
    st.write(f"Kurtosis of daily returns for a {period} period: {kurtosis}")
    st.write(f"The skewness of daily returns for a {period} period: {skewness}")
    st.write(f"Average daily return for the past {period} period: {av_return*100:.2f}%")
    st.write(f"Average daily loss of losing days for the past {period} period: {av_loss*100:.2f}%")
    st.write(f"Average daily gain of winning days for the past {period} period: {av_win*100:.2f}%")
    st.write(f"The daily win/loss ratio for the past {period} period: {win_loss_ratio}")
    st.write(f"Maximum drawdown for the past {period} period: {max_drawdown*100:.2f}%")
    st.write(f"Expected shortfall with {int((confidence_interval) * 100)}% confidence level and {frequency} frequency: {es_adjusted*100:.2f}%")
    st.write(f"Value at Risk with {int((confidence_interval) * 100)}% confidence level and {frequency} frequency: {VaR_adjusted*100:.2f}%")

    st.subheader("Plots:")

    graph_opt3 = st.selectbox("Please select the single graph you would like to depict:", ["Cumulative Returns vs Benchmark", "Cumulative Returns vs Benchmark (Log Scaled)", "Return Quantiles", "Underwater Plot", "Worst 5 Drawdown Periods", "Portfolio Earnings", "Distribution of Monthly Returns", "Strategy - Monthly Returns(%)", "Rolling Beta to Benchmark", "Rolling Sortino (6-Months)", "Rolling Sharpe (6-Months)", "Rolling Volatility (6-Months)", "Rolling Summary", "EOY Returns vs Benchmark", "Full Report" ])

    if graph_opt3 == "Cumulative Returns vs Benchmark":
        fig_cumulative_returns = qs.plots.returns(stock, benchmark, show= False, match_volatility=True)
        st.pyplot(fig_cumulative_returns)

    if graph_opt3 == "Cumulative Returns vs Benchmark (Log Scaled)":
        fig_log_returns = qs.plots.log_returns(stock, benchmark, show=False, match_volatility=True)
        st.pyplot(fig_log_returns)

    if graph_opt3 == "Return Quantiles":
        fig_distribution = qs.plots.distribution(stock, show=False)
        st.pyplot(fig_distribution)

    if graph_opt3 == "Underwater Plot":
        fig_drawdown = qs.plots.drawdown(stock, show=False)
        st.pyplot(fig_drawdown)

    if graph_opt3 == "Worst 5 Drawdown Periods":
        fig_drawdown_periods = qs.plots.drawdowns_periods(stock, show=False)
        st.pyplot(fig_drawdown_periods)

    if graph_opt3 == "Portfolio Earnings":
        fig_earnings = qs.plots.earnings(stock, show=False)
        st.pyplot(fig_earnings)

    if graph_opt3 == "Distribution of Monthly Returns":
        fig_histogram = qs.plots.histogram(stock, show=False)
        st.pyplot(fig_histogram)

    if graph_opt3 == "Strategy - Monthly Returns(%)":
        fig_heatmap = qs.plots.monthly_heatmap(stock, show=False)
        st.pyplot(fig_heatmap)

    if graph_opt3 == "Rolling Beta to Benchmark":
        fig_rolling_beta = qs.plots.rolling_beta(stock, benchmark, show=False)
        st.pyplot(fig_rolling_beta)

    if graph_opt3 == "Rolling Sortino (6-Months)":
        fig_rolling_sortino = qs.plots.rolling_sortino(stock, benchmark, show=False)
        st.pyplot(fig_rolling_sortino)

    if graph_opt3 == "Rolling Sharpe (6-Months)":
        fig_rolling_sharpe = qs.plots.rolling_sharpe(stock, benchmark, show=False)
        st.pyplot(fig_rolling_sharpe)
    
    if graph_opt3 == "Rolling Volatility (6-Months)":
        fig_rolling_volatility = qs.plots.rolling_volatility(stock, benchmark, show=False)
        st.pyplot(fig_rolling_volatility)
    
    if graph_opt3 == "Rolling Summary":
        fig_snapshot = qs.plots.snapshot(stock, show=False)
        st.pyplot(fig_snapshot)

    if graph_opt3 == "EOY Returns vs Benchmark":
        fig_yearly_returns = qs.plots.yearly_returns(stock, benchmark, show=False)
        st.pyplot(fig_yearly_returns)

    if graph_opt3 == "Full Report":
        # Generate the QuantStats HTML report
        html_file = "quantstats_report.html"
        qs.reports.html(stock, benchmark, output=html_file)

        # Display the report in Streamlit
        st.subheader("QuantStats Report")
        with open(html_file, "r", encoding="utf-8") as f:
            html_report = f.read()

        st.components.v1.html(html_report, height=1000, width=1200, scrolling=True)
############################################################################################################################################

############################################################################################################################################
if selection == "Opt4: Advanced portfolio analysis":
    
    st.header("Portfolio Analysis")

    st.write("This portfolio analysis section allows us to compare and analyse a defined portfolio's performance with benchmarks and other portfolios. The stocks are advised to be denominated in the same currency for reliable results.")

    tickers_input = st.sidebar.text_input("Here you can input the tickers of the stocks your portfolio consists (separate with commas):", value= "MSFT, AAPL, TSLA")    
    benchmark_ticker = st.sidebar.text_input("Benchmark Ticker (e.g., SPY) (yellow line on graphs):", value="SPY")
    method = st.sidebar.selectbox("Select the method of weight calculation: ", ["N.o shares (same currency):", "Stock weights: "])
    #start_date_given = st.sidebar.date_input("Select start date:", value=pd.to_datetime("2020-01-01"))
    period = st.sidebar.selectbox("Select Period:", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"])

    rebal_period = st.sidebar.selectbox("Please select rebalancing period:", ["1M", None])
    rf_rate_input = st.sidebar.number_input("Please provide the risk free rate (if desired to discount)")

    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]
    prices_df = yf.download(tickers, period="1d")["Close"].iloc[-1]
    prices = prices_df.reindex(tickers).values
    benchmark = qs.utils.download_returns(benchmark_ticker, period=period)

    prices = list(prices)

    #displaying a correlation matrix of the stocks

    st.subheader("Correlation Matrix Heatmap")
    end_date = datetime.today()
    if period == "1mo": 
        start_date = end_date - timedelta(days=30)
    elif period == "3mo":
        start_date = end_date - timedelta(days=90)
    elif period == "6mo":
        start_date = end_date - timedelta(days=180)
    elif period == "1y":
        start_date = end_date - timedelta(days=365)
    elif period == "2y":
        start_date = end_date - timedelta(days=730)
    elif period == "5y":
        start_date = end_date - timedelta(days=1825)
    elif period == "max":
        start_date = datetime(2000, 1, 1)

    #correlation
    data_correlation = yf.download(tickers, start=start_date, end=end_date)["Close"]
    returns = data_correlation.pct_change().dropna()
    correlation_matrix = returns.corr()

    #heatmap
    plt.figure(figsize=(7, 4))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix of Stock Returns")
    plt.show()

    #st.dataframe(correlation_matrix)
    st.pyplot(plt)

    if method == "N.o shares (same currency):":
        no_shares_input = st.sidebar.text_input("Here you can input the n.o shares in each stock in the right order (has to be the same currency)(separate with commas):", value= "1,2,1")
        no_shares = [int(shares.strip()) for shares in no_shares_input.split(",") if shares.strip()]
        stock_values = [price * shares for price, shares in zip(prices, no_shares)]
        portfolio_value = sum(stock_values)
        st.write("Your total portfolio value (in relative currency): ", portfolio_value)
        weights = [float(value) / float(portfolio_value) for value in stock_values]
        weights = list(weights)

    else:
        weights_input = st.sidebar.text_input("Here you can specify the weights straightaway (separate with comma, sum to 1):",value= "0.5,0.3,0.2")
        weights = [float(weight.strip()) for weight in weights_input.split(",") if weight.strip()]
        weights = list(weights)

    #tickers = list(tickers)

    portfolio_final = dict(zip(tickers, weights))
    portfolio_df = pd.DataFrame(list(portfolio_final.items()), columns=["Ticker", "Weight"])
    st.subheader("Portfolio Allocation")
    st.table(portfolio_df)

    html_file2 = "quanstsportfolio_report.html"
    fmaga = portfolio_final_qs = qs.utils.make_index(ticker_weights=portfolio_final,rebalance=rebal_period, period=period)

    #help(qs.utils.make_index)
    qs.reports.html(fmaga, benchmark,rf=rf_rate_input, output=html_file2)

    st.subheader("QuantStats Portfolio Report")
    with open(html_file2, "r", encoding="utf-8") as f:
        html_report = f.read()

    st.components.v1.html(html_report, height=1000, width=1200, scrolling=True)
############################################################################################################################################

############################################################################################################################################
if selection == "Opt5: Simple Efficient Frontier Portfolio Optimization":
    
    st.header("Portfolio Optimization with Markowitz Efficient Frontier")
    
    st.write("The Markowitz Efficient Frontier portfolio optimization method, developed by Harry Markowitz, is a cornerstone of modern portfolio theory. It identifies the set of portfolios that offer the maximum expected return for a given level of risk (or equivalently, the minimum risk for a given level of return). The method uses the mean-variance optimization framework, which considers the expected returns, variances (risks), and covariances (correlations) of asset returns.")
    st.write("By diversifying investments, the approach reduces overall portfolio risk through the selection of assets with low or negative correlations, creating the efficient frontier — a curve representing the optimal trade-offs between risk and return. Investors can then choose a portfolio from the frontier based on their risk tolerance. If there is a problem with the depiction of the results, it is likely due to the maximum weight, which should be increased.")
    tickers_input = st.sidebar.text_input("Here you can input the tickers of the stocks your portfolio consists (separate with comma):", value="MSFT, AAPL, AMZN")
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

    start_date = st.sidebar.date_input("Define the start date for which we should define the efficient portfolio:", value="2020-01-01")
    end_date = datetime.today()

    #determining the capital to be allocated
    total_value_allocateable = st.sidebar.number_input("Capital to be allocated between stocks (currency is dollars by default):", value=3000)

    #determining the maximum weight of an optimal stock
    max_weight = st.sidebar.number_input("The maximum weight assigned to each asset in the optimal portfolio:",value=0.7 )

    #inputting the risk-free rate
    rf_rate = st.sidebar.number_input("Provide the risk_free rate for the period for more realistic optimization (e.g.: 0.03 = 3%) (use U.S. Treasury yield rates for the relevant period e.g.: 1y, 5y, 10y):",value=0.04 )

    #downloading the dataframe for the calculations
    df = yf.download(tickers, start= start_date, end=end_date)["Close"]

    mu = expected_returns.mean_historical_return(df)
    S = risk_models.sample_cov(df)

    sorted_tickers = sorted(tickers)  # Sort tickers alphabetically
    sorted_mu = [mu[tickers.index(ticker)] for ticker in sorted_tickers]
    
    returns_df = pd.DataFrame({
        "Ticker": sorted_tickers,
        "Expected Return": sorted_mu})

    # Display the DataFrame as a table in Streamlit
    st.subheader("Expected Returns for Stocks")
    st.table(returns_df)

    #Optimizing for maximum Sharpe ratio
    ef = EfficientFrontier(mu, S, weight_bounds=(0,max_weight))
    raw_weights = ef.max_sharpe(risk_free_rate=rf_rate)
    cleaned_weights = ef.clean_weights()
    ef.save_weights_to_file("weights.csv")
    performance = ef.portfolio_performance(risk_free_rate=rf_rate, verbose=False)
    expected_return_performance = performance[0]
    annual_volatility_performance = performance[1]
    sharpe_ratio_performance = performance[2]

    st.subheader("Main optimal portfolio statistics")
    st.write(f"Expected annual return: {expected_return_performance*100:.2f}%")
    st.write(f"Annual volatility: {annual_volatility_performance*100:.2f}%")
    st.write(f"Sharpe ratio: {sharpe_ratio_performance}")

    # Convert weights dictionary to a pandas DataFrame
    weights_df = pd.DataFrame.from_dict(cleaned_weights, orient="index", columns=["Weight"])

    # Reset index to make the stock names a column
    weights_df.reset_index(inplace=True)
    weights_df.rename(columns={"Index": "Stock"}, inplace=True)

    # Display the DataFrame in a tabular format
    st.subheader("Portfolio Weights")
    st.table(weights_df)  # Interactive table with sorting and scrolling

    #piechart
    # Extract tickers and weights with non-zero values
    tickers_pie = [ticker for ticker, weight in cleaned_weights.items() if weight > 0]
    weights_pie = [weight for weight in cleaned_weights.values() if weight > 0]

    # Ensure tickers and weights are aligned
    if len(tickers_pie) != len(weights_pie):
        st.write("Error: Mismatch between tickers and weights!")

    # Define a function to format the percentage labels
    def autopct_format(pct):
        return f'{pct:.1f}%' if pct > 0 else ''

    # Plot the pie chart
    fig2, ax2 = plt.subplots()
    ax2.pie(
        weights_pie,
        labels=tickers_pie,
        autopct=autopct_format,
        startangle=90,
    )
    ax2.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
    ax2.set_title("Portfolio distribution")
    st.pyplot(fig2)

    latest_prices = get_latest_prices(df)

    da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=total_value_allocateable)
    allocation, leftover = da.greedy_portfolio()

    allocation_df = pd.DataFrame.from_dict(allocation, orient="index", columns=["N.o. shares"])
    allocation_df.reset_index(inplace=True)
    allocation_df.rename(columns={"index": "Stock"}, inplace=True)
    
    st.subheader("Shares to be allocated")
    st.table(allocation_df)

    #st.write("Discrete allocation: ", allocation)
    st.write("Funds remaining: ${:.2f}".format(leftover))

    #creating the efficient frontier
    ef_curve = EfficientFrontier(mu, S, weight_bounds=(0,1))

    #generating points for the efficient frontier
    fig1, ax = plt.subplots()

    #using the annotation for the dots
    for i, txt in enumerate(df.columns):
        stock_volatility = S.iloc[i, i] ** 0.5  # Volatility (standard deviation)
        stock_return = mu[i]  # Expected return
        ax.scatter(stock_volatility, stock_return, marker="o", label=None)  # Plot individual asset
        ax.annotate(txt, (stock_volatility, stock_return), textcoords="offset points", xytext=(5,5), ha='center')

    #plotting the graph
    plotting.plot_efficient_frontier(ef_curve, ax = ax)
    ax.scatter(annual_volatility_performance, expected_return_performance , marker="*", s=100, c="r", label='Optimal Portfolio')
    ax.set_xlabel("Volatility (Standard Deviation)")
    ax.set_ylabel("Expected Return")
    ax.legend()

    st.subheader("Markowicz Efficient Frontier")
    st.pyplot(fig1)
############################################################################################################################################

############################################################################################################################################
if selection == "Opt6: Black-Litterman Efficient Frontier Portfolio Optimization": 

    #opening
    st.header("Black-Litterman Portfolio Optimization")
    
    st.write("The Black-Litterman portfolio optimization method is a framework that combines modern portfolio theory (e.g., the Markowitz efficient frontier) with investor views to generate an improved estimate of expected returns. It starts with a baseline of market-implied returns derived from the market capitalization weights and the covariance matrix of asset returns. Investors can then incorporate their subjective views (absolute or relative) about certain assets' returns, which are blended with the market's baseline using Bayesian principles.")

    st.write("This approach helps overcome issues like over-concentration in portfolios and instability in traditional mean-variance optimization, producing more balanced and realistic portfolios that reflect both market data and investor preferences. If there is a problem with the depiction of the results, it is likely due to the maximum weight, which should be increased.")

    #getting the tickers
    tickers_input = st.sidebar.text_input("Here you can input the tickers of the stocks your portfolio consists (separate with comma):", value="MSFT, AAPL, AMZN")
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

    #benchmark
    benchmark = st.sidebar.text_input("Here you can define the benchmark: ", value="SPY")

    #getting the dates for the calculations
    start_date = st.sidebar.date_input("Define the start date for which we should define the efficient portfolio:", value="2020-01-01")
    end_date = datetime.today()

    #getting risk free rate
    rf_rate = st.sidebar.number_input("Provide the risk_free rate for the period for more realistic optimization (e.g.: 0.03 = 3%) (use U.S. Treasury yield rates for the relevant period e.g.: 1y, 5y, 10y):",value=0.04 )

    #determining the capital to be allocated
    total_value_allocateable = st.sidebar.number_input("Capital to be allocated between stocks (currency is dollars by default):", value=3000 )

    #getting maximum weight for a stock
    max_weight = st.sidebar.number_input("The maximum weight assigned to each asset in the optimal portfolio:", value=0.5 )

    #downloading stock data
    portfolio = yf.download(tickers, start_date, end_date)['Close']
    benchmark = yf.download(benchmark, start_date, end_date)['Close']

    #downloading market capitalization for each stock in the portfolio
    mcaps = {}
    for t in tickers: 
        stock = yf.Ticker(t)
        mcaps[t] = stock.info["marketCap"]

    #calculating implied market returns
    S = risk_models.CovarianceShrinkage(portfolio).ledoit_wolf()
    delta = black_litterman.market_implied_risk_aversion(benchmark, risk_free_rate=rf_rate)
    st.write("Delta: ", delta)

    #heatmap for the covariant correlations
    fig1, ax1, = plt.subplots(figsize=(8,6))
    sns.heatmap(S.corr(), cmap = 'coolwarm', annot=True, ax=ax1)
    st.subheader("Covariant Correlation Heatmap")
    st.pyplot(fig1)

    #generating the prior 
    market_prior = black_litterman.market_implied_prior_returns(mcaps, delta, S, risk_free_rate=rf_rate)
    st.subheader("Market prior expected returns")
    st.table(market_prior)

    #horizontal barchart for visualization
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    market_prior.plot.barh(ax=ax2)
    ax2.set_title("Market Prior Compensation")
    ax2.set_xlabel("Percentage")
    ax2.set_ylabel("Stocks")
    st.pyplot(fig2)

    #creating a dictionary to integrate views
    st.sidebar.subheader("The views on each stock(e.g., 0.10 means the stock will move up 10% in 1 year):")
    views = {}
    for ticker in tickers:
        views[ticker] = st.sidebar.number_input(f"Expected return for {ticker}:", value=0.2, format="%.2f")
    # Display the resulting dictionary
    bl = BlackLittermanModel(S, pi=market_prior, absolute_views=views)

    confidence_level = st.sidebar.slider(
    "Select Confidence Level:",
    min_value=0.80,
    max_value=0.99,
    value=0.95,  # Default to 95%
    step=0.01)
    # Calculate the z-score corresponding to the confidence level
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    # Calculate confidence intervals using log returns
    intervals = []
    # Compute confidence intervals for each stock in the portfolio
    for ticker in tickers:
        # Calculate log returns
        stock_simple_returns = portfolio[ticker].pct_change().dropna()
        # Calculate mean (from views) and standard deviation (from log returns)
        mean_return = views[ticker]
        std_dev = stock_simple_returns.std()
        # Calculate confidence interval using z-score
        lower_bound = mean_return - z_score * std_dev
        upper_bound = mean_return + z_score * std_dev
        # Append to intervals list
        intervals.append((lower_bound, upper_bound))
    # Display the confidence intervals
    intervals_df = pd.DataFrame(intervals, columns=["Lower Bound", "Upper Bound"], index=tickers)
    st.subheader("Confidence intervals")
    st.table(intervals_df)

    #calculating omega
    variances = []
    for lb, ub in intervals:
        sigma = (ub-lb)/2
        variances.append(sigma ** 2)
    omega = np.diag(variances)

    bl_2 = BlackLittermanModel(S, pi="market", market_caps=mcaps, risk_aversion=delta, absolute_views=views, omega=omega)
    ret_bl = bl_2.bl_returns()

    #passing all of this into a dataframe
    rets_df = pd.DataFrame([market_prior, ret_bl, pd.Series(views)], index=["Prior", "Posterior", "Views"]).T
    st.subheader("Market prior, posterior, and absolute view expected returns table")
    st.table(rets_df)

    #the other figure
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    rets_df.plot.bar(ax=ax3)
    ax3.set_title("Comparing Compensations Barchart")
    st.pyplot(fig3)

    #allocating this to a portfolio
    S_bl = bl.bl_cov()

    ef = EfficientFrontier(ret_bl, S_bl, weight_bounds=(0,max_weight))
    #ef.add_objective(objective_functions.L2_reg)
    raw_weights = ef.max_sharpe(risk_free_rate=rf_rate)
    clean_weights = ef.clean_weights()
    ef.save_weights_to_file("weights2.csv")
    performance = ef.portfolio_performance(risk_free_rate=rf_rate, verbose=False)
    expected_return_performance = performance[0]
    annual_volatility_performance = performance[1]
    sharpe_ratio_performance = performance[2]

    st.subheader("Main optimal portfolio statistics")
    st.write(f"Expected annual return: {expected_return_performance*100:.2f}%")
    st.write(f"Annual volatility: {annual_volatility_performance*100:.2f}%")
    st.write(f"Sharpe ratio: {sharpe_ratio_performance}")

    # Convert weights dictionary to a pandas DataFrame
    weights_df = pd.DataFrame.from_dict(clean_weights, orient="index", columns=["Weight"])

    # Reset index to make the stock names a column
    weights_df.reset_index(inplace=True)
    weights_df.rename(columns={"Index": "Stock"}, inplace=True)

    # Display the DataFrame in a tabular format
    st.subheader("Optimal Portfolio Weights")
    st.table(weights_df)  # Interactive table with sorting and scrolling

    #piechart
    # Extract tickers and weights with non-zero values
    tickers_pie = [ticker for ticker, weight in clean_weights.items() if weight > 0]
    weights_pie = [weight for weight in clean_weights.values() if weight > 0]

    # Ensure tickers and weights are aligned
    if len(tickers_pie) != len(weights_pie):
        st.write("Error: Mismatch between tickers and weights!")

    # Define a function to format the percentage labels
    def autopct_format(pct):
        return f'{pct:.1f}%' if pct > 0 else ''

    # Plot the pie chart
    fig5, ax5 = plt.subplots()
    ax5.pie(
        weights_pie,
        labels=tickers_pie,
        autopct=autopct_format,
        startangle=90,
    )
    ax5.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
    ax5.set_title("Portfolio distribution")
    st.pyplot(fig5)

    latest_prices = get_latest_prices(portfolio)

    da = DiscreteAllocation(clean_weights, latest_prices, total_portfolio_value=total_value_allocateable)
    allocation, leftover = da.greedy_portfolio()

    allocation_df = pd.DataFrame.from_dict(allocation, orient="index", columns=["N.o. shares"])
    allocation_df.reset_index(inplace=True)
    allocation_df.rename(columns={"index": "Stock"}, inplace=True)
    st.subheader(f"Shares to be allocated out of ${total_value_allocateable}")
    st.table(allocation_df)

    #st.write("Discrete allocation: ", allocation)
    st.write("Funds remaining: ${:.2f}".format(leftover))

    #creating the efficient frontier
    ef_curve = EfficientFrontier(ret_bl, S_bl, weight_bounds=(0,max_weight))
    #ef_curve.add_objective(objective_functions.L2_reg)

    #generating points for the efficient frontier
    fig6, ax6 = plt.subplots()

    #using the annotation for the dots
    for i, txt in enumerate(portfolio.columns):
        stock_volatility = S_bl.iloc[i, i] ** 0.5  # Volatility (standard deviation)
        stock_return = ret_bl[i]  # Expected return
        ax6.scatter(stock_volatility, stock_return, marker="o", label=None)  # Plot individual asset
        ax6.annotate(txt, (stock_volatility, stock_return), textcoords="offset points", xytext=(5,5), ha='center')

    #plotting the graph
    plotting.plot_efficient_frontier(ef_curve, ax = ax6)
    ax6.scatter(annual_volatility_performance, expected_return_performance , marker="*", s=100, c="r", label='Optimal Portfolio')
    ax6.set_xlabel("Volatility (Standard Deviation)")
    ax6.set_ylabel("Expected Return")
    ax6.legend()

    st.subheader("Markowicz Efficient Frontier")
    st.pyplot(fig6)
############################################################################################################################################
