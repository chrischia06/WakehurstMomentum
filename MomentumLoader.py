import numpy as np
import pandas as pd

def clean(df2):
    df = df2.copy()
    try:
        df['Price'] = df['Price'].apply(lambda x: x.replace(",",""))
        df['High'] = df['High'].apply(lambda x: x.replace(",",""))
        df['Open'] = df['High'].apply(lambda x: x.replace(",",""))
        df['Low'] = df['Low'].apply(lambda x: x.replace(",",""))
        df[['Price','High','Open','Low']] = df[['Price','High','Open','Low']].astype('float64')
    except:
        pass
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df



def transformDF(df, swap=False):
    '''
    Inputs: 
    Expects a dataframe with Columns: Price, Date

    Outputs: 
    Returns Two dataframes with all generated features: Seasonality (Day, Month, Weekday), Returns
    df2 : dataframe for Tree methods
    df3: dataframe for normal methods
    '''
    normalizeFeats = []
    df2 = df.copy()
    if swap == True:
        df2['Price'] = df2['Rate']
        df2 = df2.drop('Rate', axis=1)

    #Price/ Returns
    if 'Price' in list(df2.columns):
        df2['returns'] = df2['Price'] / df2['Price'].shift(1)
        df2.loc[0,'returns'] = 1
        df2['log_returns'] = np.log(df2['returns'])
        df2['returns'] -=1
        df2['Price'] = np.log(df2['Price'])
    #Volatility
    if 'High' in list(df2.columns) and 'Low' in df2.columns:
        # df2['IntradayRange'] = np.log(df2['High']/df2['Low']) #intraday range
        df2[['High','Low']] = np.log(df2[['High','Low']]) #log transform
        #normalizeFeats = ['Price','High','Low']
    else:
        pass
        #normalizeFeats = ['Price']
    if 'Open' in list(df2.columns):
        df2['Open'] = np.log(df2['Open'])
    
    #Dates
    df3 = df2.copy()
    #Trees can support categories as numbers    
    df2['month'] = df2['Date'].dt.month
    df2['day']  = df2['Date'].dt.day
    df2['weekday'] = df2['Date'].dt.dayofweek
    
    #one hot encode dates, leave one out for Neural Networks/ Regression
    
    df3 = pd.concat([df3,pd.get_dummies(df2['month'],prefix="month",drop_first=True)],axis=1)
    df3 = pd.concat([df3,pd.get_dummies(df2['day'],prefix="day",drop_first=True)],axis=1)
    df3 = pd.concat([df3,pd.get_dummies(df2['weekday'],prefix="weekday",drop_first=True)],axis=1)
    
    #Drop nonstationary features
    try:
        df3 = df3.drop(['returns'],axis=1)
    except:
        pass
    print("ADDED FEATURES - COMPLETE")
    print("--------------------")
    return df2, df3, normalizeFeats



feats = ['Price', 'High', 'Low', 'log_returns']
feats2 = ['log_returns']
def addLaggedFeats(df2, df3, LAGS):
    df4 = df2.copy()
    df5 = df3.copy()
    try:
        for t in range(1,LAGS + 1):
            df4[[x +"t_nm"+str(t) for x in feats]] = df2[feats].shift(t)    
        for t in range(1,LAGS + 1):
            df5[[x +"t_nm"+str(t) for x in feats2]] = df3[feats2].shift(t)
        df4["vol_returns_"+str(LAGS)] = df2['log_returns'].rolling(LAGS).std()
        df5["vol_returns_"+str(LAGS)] = df2['log_returns'].rolling(LAGS).std()
    except:
        pass
    return df4.iloc[LAGS:].reset_index(drop=True), df5.iloc[LAGS:].reset_index(drop=True)    
