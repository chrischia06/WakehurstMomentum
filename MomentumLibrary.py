import json
import plotly
import ipywidgets as widgets
import statsmodels.api as sm

from MomentumML import *
from MomentumBacktester import *
from MomentumLoader import *
from scraper_utility import *



def statistics(df):
    print("HISTORICAL DATA STATISTICS")
    print("--------------------")
    print(f"Number of days: {df.shape[0]}")
    print("Data Start:",df['Date'].iloc[0])
    print("Data End:", df['Date'].iloc[-1])
    print("--------------------")
    print("Tests for Normality")
    print("--------------------")
    print(f"Mean Returns: {(df['returns'].mean()):.6f}%")
    print(f"Std Dev of Returns (Vol): {(df['returns'].std()):.6f}")
    print(f"Skew of Returns (Vol): {(df['returns'].skew()):.6f}")
    print(f"Kurtosis of Returns (Normal 3): {(df['returns'].kurtosis()):.6f}")
    print("--------------------")
    print(f"Mean Log-Returns: {(df['log_returns'].mean()):.6f}")
    print(f"Std Dev of Log-Returns (Vol): {(df['log_returns'].std()):.6f}")
    print(f"Skew of Log Returns: {df['log_returns'].skew()}")
    print(f"Kurtosis of Log Returns: {df['log_returns'].kurtosis()}")
    print("--------------------")
    print("Tests for Stationarity:")
    print("--------------------")
    print(f"Test for stationarity of returns: p-value - {adfuller(df['returns'].iloc[1:])[1]:.5f}")
    print(f"Test for stationarity of price: p-value - {adfuller(df['Price'].iloc[1:])[1]:.5f}")
    print(f"Test for stationarity of log_returns: p-value - {adfuller(df['log_returns'].iloc[1:])[1]:.5f}")
    print("--------------------")
    fig, ax = plt.subplots(figsize=(30,5),ncols=3)
    sns.distplot(df['log_returns'],ax=ax[0])
    ax[0].set_title('Log Returns')
    sns.distplot(df['returns'],ax=ax[1])
    ax[1].set_title('Returns')
    sm.graphics.tsa.plot_pacf(df['log_returns'].values.squeeze(),lags=list(range(1,60,1)),ax=ax[2])
    return ax

try:
    with open("../assets.json") as f:
        assets = json.load(f)
except:
    try:
        with open("assets.json") as f:
            assets = json.load(f)
    except:
        print("Failed to load assets file")


datasets = widgets.Dropdown(
    options=list(assets.keys()),
    value='Brent Oil Futures',
    description='Asset:',
    disabled=False,
)

def makeWidget(df3, exog_name):
    featsToUseWidget = widgets.SelectMultiple(
        options=df3.drop('Date',axis=1).columns,
        value=[exog_name],
        #rows=10,
        description='Variables',
        disabled=False
    )
    return featsToUseWidget