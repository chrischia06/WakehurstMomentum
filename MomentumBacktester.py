import numpy as np
import pandas as pd
import empyrical
from scipy.stats import binom
import plotly.graph_objects as go
from datetime import datetime, date

def backtestLogger(models, data_tree, data, param, swap, ID, write=False,verbose=False):
    print("BACKTEST SEARCHER")
    print("--------------------")
    results = []
    if param == 'Rate':
        param = 'Price'
    if param == 'direction':
        for modelToUse in ['cbc','logit','dnn','ensemble','gpc']:
            for conf in [-1,0, 0.25, 0.5, 0.75,0.9]:
                print(f"PARAMS - Model: {modelToUse}, Confidence:{conf}")
                transaction_costs = 0
                backtestResults,tradeBook = backtest(models, data_tree, data, transaction_costs, modelToUse, param, swap, starting_cap = 10000,  confidence=conf, longLimit=1, shortLimit=-1, reinvest=True,write=False, path = "")
                fig, fig2, summary = diagnoseBackTest(backtestResults,tradeBook,data_tree['date_test'], swap,10000, verbose=verbose)
                summary['model'] = modelToUse
                summary['confidence_level'] = conf
                results += [summary]

        for conf in [-1,0, 0.25, 0.5, 0.75,0.9]:
            print(f"PARAMS - Model: xgbc, Confidence:{conf}")
            transaction_costs = 0
            backtestResults,tradeBook = backtest(models, data_tree, data,transaction_costs, 'xgbc', param, swap, starting_cap = 10000,confidence=0, longLimit=1,shortLimit=-1, reinvest=True, write=False, path ="")
            fig, fig2, summary = diagnoseBackTest(backtestResults,tradeBook,data_tree['date_test'], swap,10000,verbose=verbose)
            summary['model'] = 'xgbc'
            summary['confidence_level'] = conf
            results += [summary]

    elif param in ['log_returns','Price', 'returns']:
        for modelToUse in ['cbr','linear_reg','dnn','ensemble','xgbr','gpr']:
                print(f"PARAMS - Model: {modelToUse}")
                transaction_costs = 0
                backtestResults,tradeBook = backtest(models, data_tree, data, transaction_costs, modelToUse, param, swap,starting_cap = 10000,  confidence=0, longLimit=1, shortLimit=-1,reinvest=True, write=False, path = "")
                fig, fig2, summary = diagnoseBackTest(backtestResults,tradeBook,data_tree['date_test'], swap,10000, verbose=verbose)
                summary['model'] = modelToUse
                summary['confidence_level'] = 0
                results += [summary]
    
    results = pd.json_normalize(results)
    results['trainPerformanceID'] = ID
    if write == True:
        results.to_csv(f"backtestLog-{str(datetime.now())[:19]}).csv",index=False)
    results = results.sort_values('totalPNL',ascending=False).reset_index(drop=True)

    print("--------------------")
    return isSwap(results, swap)

def isSwap(results,swap):
    results2 = results.copy()
    if swap == True:

        results2[['totalPNL','meanPNL','worstTrade','bestTrade']] *= 100
    return results2

def isSwap2(tradeBook,swap):
    tradeBook2 = tradeBook.copy()
    if swap == True:
        tradeBook2.columns = ['entryRate','dv01','entryDate','entrySignal','entryReason','exitDate','exitRate','exitReason']
        tradeBook2['entryReason'] = tradeBook2['entryReason'].apply(lambda x: swapEntryReason(x))
        tradeBook2['exitReason'] = tradeBook2['exitReason'].apply(lambda x: swapExitReason(x))
    return tradeBook2

def swapEntryReason(x):
    reasonsMap = {'Entering/Expanding SHORT':"Entering position in Receivers", 
    'Entering/Expanding LONG': "Entering  position in Payers", 
    'Entering Reduced position (SHORT)': "Reduced exposure (Receivers)",
    "Entering reduced position (LONG)": "Reduced exposure (Payers)"}
    return reasonsMap[x]

def swapExitReason(x):
    if x:
        reasonsMap = {'Reducing position size (SHORT)':"Closing Receivers", 
        'CHANGE IN FORECAST DIRECTION (now INCREASE)': "CHANGE IN FORECAST DIRECTION (now INCREASE)", 
        'CHANGE IN FORECAST DIRECTION (to DECREASE)': "Reduced exposure (Receivers)",
        "Reducing position size (LONG)": "Closing Payers",
        "Not Enough Model Confidence": "Not Enough Model Confidence",
        }
        return reasonsMap[x]
    else:
        return np.nan




def backtest(models, data_tree, data, transaction_costs, modelToUse, param, swap,starting_cap = 10000,  confidence=0, longLimit=1, shortLimit=-1,reinvest=True, write=True, path = ""):
    df3 = data_tree['X_test'].copy().reset_index(drop=True)
    df3['signal'] = np.nan
    df3['cash'] = 0
    df3['value'] = 0
    df3['units'] = 0
    df3['traded'] = 0
    
    #add data
    df3['Date'] = data_tree['date_test'].values
    dates = df3['Date'].values
    #add signals to the dataframe
    if param == 'direction':
        if modelToUse in ['xgbc', 'cbc']:
            df3['signal'] = models[modelToUse].predict_proba(data_tree['X_test'])[:,1]
        elif modelToUse in ['logit','gpc']:
            df3['signal'] = models[modelToUse].predict_proba(data['X_test'])[:,1]
        elif modelToUse in ['dnn']:
            df3['signal'] = models[modelToUse].predict(data['X_test']).reshape(-1)
        elif modelToUse in ['ensemble']:
            df3['signal'] = models[modelToUse].predict_proba(data_tree['ensemble_test'])[:,1]
    elif param in ['log_returns','Price','returns']:
        if modelToUse in ['xgbr', 'cbr']:
            df3['signal'] = models[modelToUse].predict(data_tree['X_test'])
        elif modelToUse in ['linear_reg','dnn']:
            df3['signal'] = models[modelToUse].predict(data['X_test']).reshape(-1)
        elif modelToUse in ['ensemble']:
            df3['signal'] = models[modelToUse].predict(data_tree['ensemble_test'])
    signals = list(df3.loc[(df3['Date'] >= dates[0]), 'signal'].values)
    
    #set starting cash, book value to starting_cap
    df3.loc[df3['Date'] == dates[0], ['cash','value']] = starting_cap
    
    #initialise trade book
    tradeBook = {}
    tradeBook['longs'] = []
    tradeBook['shorts'] = []
    
    #format of a trade {"entryPrice":, "units", "exitPrice", "entryDate","exitDate", "reason"}
    
    #loop through and backtest
    for i in range(len(dates)):
        #get todays open price
        price = df3.loc[df3['Date']==dates[i],'Price'].values[0]
        #your value for the date is your cash + position * units at open
        df3.loc[df3['Date']==dates[i],'value'] = df3.loc[df3['Date']==dates[i],'cash'].values[0] + df3.loc[df3['Date']==dates[i],'units'].values[0] * price
        #this converts into a long/short signal between [-1, 1]
        if param == 'direction':
            proportion = 2 * signals[i] - 1
            if confidence == -1:
                proportion = 2 * (signals[i] - 0.5) * 1 -1 
        elif param in ['log_returns', 'returns']:
            proportion = 2 * (signals[i] > 0) * 1 - 1
        elif param == 'Price':
            proportion = 2 * (signals[i] > price) * 1 - 1
        
        #If confidentadjust position size based on limits on long/short and set target new position
        if abs(proportion) >= confidence:
            if proportion > 0:
                proportion = min(longLimit, proportion)
            else:
                proportion = max(shortLimit, proportion)
            if reinvest == True:
                new_units = proportion * df3.loc[(df3['Date'] == dates[i]), 'value'].values[0] / price
            else:
                new_units = proportion * starting_cap / price
        else: #otherwise, close positions
            new_units = 0
    
        current_units = df3.loc[(df3['Date'] == dates[i]), 'units'].values[0]
        df3.loc[(df3['Date'] == dates[i]), 'cash'] -= (new_units - current_units) * price
        df3.loc[(df3['Date'] == dates[i]), 'units'] = new_units
        
        if current_units > new_units: #this is either a short, or close longs + short
            df3.loc[(df3['Date'] == dates[i]), 'traded'] = -1
            if new_units  <= 0:
                for j in range(len(tradeBook['longs'])): #close all longs
                    if tradeBook['longs'][j]['exitDate'] == None:
                        tradeBook['longs'][j]['exitDate'] = dates[i]
                        tradeBook['longs'][j]['exitPrice'] = price
                        if new_units == 0:
                            tradeBook['longs'][j]['exitReason'] = "Not Enough Model Confidence"
                        elif new_units < 0:
                            tradeBook['longs'][j]['exitReason'] = "CHANGE IN FORECAST DIRECTION (to DECREASE)"
                if new_units < 0: #add a short
                    tradeBook['shorts'] += [{"entryPrice":price, 
                                            "units":(new_units-min(0,current_units)),
                                            "entryDate":dates[i],
                                            "entrySignal":(2*signals[i]-1),
                                            "entryReason":"Entering/Expanding SHORT",
                                            "exitDate":None,
                                            "exitPrice":None,
                                            "exitReason":None}] 

            else: #current_units < 0; #expand short size
                temp = current_units
                for j in range(len(tradeBook['longs'])):
                    if tradeBook['longs'][j]['exitDate'] == None:
                        if temp >= new_units:
                            tradeBook['longs'][j]['exitDate'] = dates[i]
                            tradeBook['longs'][j]['exitPrice'] = price
                            tradeBook['longs'][j]['exitReason'] = "Reducing position size (LONG)"
                            temp -= tradeBook['longs'][j]['units']
                        else:
                            break
                if new_units - temp > 0:
                    tradeBook['longs'] += [{"entryPrice":price, 
                                    "units":(new_units-temp),
                                    "entryDate":dates[i],
                                    "entrySignal":(2*signals[i]-1),
                                    "entryReason":"Entering reduced position (LONG)",
                                    "exitDate":None,
                                    "exitPrice":None,
                                    "exitReason":None}]
    
                
        elif current_units < new_units: # this is either a long, or close shorts + long
            df3.loc[(df3['Date'] == dates[i]), 'traded'] = 1
            if new_units >= 0: 
                for j in range(len(tradeBook['shorts'])): #close any shorts
                    if tradeBook['shorts'][j]['exitDate'] == None:
                        tradeBook['shorts'][j]['exitDate'] = dates[i]
                        tradeBook['shorts'][j]['exitPrice'] = price
                        if new_units == 0:   
                            tradeBook['shorts'][j]['exitReason'] = "Not Enough Model Confidence"
                        if new_units > 0:   
                            tradeBook['shorts'][j]['exitReason'] = "CHANGE IN FORECAST DIRECTION (now INCREASE)"
                if new_units > 0: # add a long
                    tradeBook['longs'] += [{"entryPrice":price, 
                                                "units":new_units-max(0, current_units),
                                                "entryDate":dates[i],
                                                "entrySignal":(2*signals[i]-1),
                                                "entryReason":"Entering/Expanding LONG",
                                                "exitDate":None,
                                                "exitPrice":None,
                                                "exitReason":None}]
            else: #current < new_units < 0
                temp = current_units
                for j in range(len(tradeBook['shorts'])):
                    if tradeBook['shorts'][j]['exitDate'] == None:
                        if temp  <= new_units:
                            tradeBook['shorts'][j]['exitDate'] = dates[i]
                            tradeBook['shorts'][j]['exitPrice'] = price
                            tradeBook['shorts'][j]['exitReason'] = "Reducing position size (SHORT)"
                            temp -= tradeBook['shorts'][j]['units']
                        else:
                            break
                if new_units - temp > 0:
                    tradeBook['shorts'] += [{"entryPrice":price, 
                                        "units":(new_units-temp),
                                        "entryDate":dates[i],
                                        "entrySignal":(2*signals[i]-1),
                                        "entryReason":"Entering Reduced position (SHORT)",
                                        "exitDate":None,
                                        "exitPrice":None,
                                        "exitReason":None}]
                
        #carry forward balance
        if i < len(dates)-1:
            df3.loc[(df3['Date'] == dates[i+1]),['cash','units']] = df3.loc[(df3['Date'] == dates[i]), ['cash','units']].values[0]
    df3['backtest_returns'] = df3['value'] / df3['value'].shift(1)

    tradeBook = pd.concat([pd.json_normalize(tradeBook['longs']),pd.json_normalize(tradeBook['shorts'])],axis=0)
    if tradeBook.shape[0] > 0:
        tradeBook = tradeBook.sort_values('entryDate').reset_index(drop=True)

    if write == True:
        backtestInterval = df3.loc[df3['Date'] >= data_tree['date_test'].values]
        df3.to_csv(f"{path}/momentum_{str(datetime.now())[:19]})_RAW.csv",index=False)
        tradeBook.to_csv(f"{path}/momentum_{str(datetime.now())[:19]})_TRADEBOOK.csv",index=False)
    return df3, tradeBook

def diagnoseBackTest(backtestResults, tradeBook, date_test, swap, starting_cap, verbose=False):
    if tradeBook.shape[0] == 0:
        print("NO TRADES MADE")
        return None, None, {}
    backtestInterval = backtestResults.loc[backtestResults['Date'] >= date_test.iloc[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pd.to_datetime(backtestInterval['Date'].iloc[1:-1]), 
                             y=backtestInterval['backtest_returns'].iloc[1:-1].cumprod(),
                             name='Backtested Strategy Performance',
                             mode='markers+lines',
                            marker={"size":12},
                            marker_color=backtestInterval['traded']))
    fig.add_trace(go.Scatter(x=backtestInterval['Date'].iloc[:-1], 
                             y=np.exp((backtestInterval['log_returns']).iloc[:-1].cumsum()), 
                             name='Benchmark returns'))
    fig.update_layout(title="Backtested Strategy vs Buy and Hold Returns (Cumulative)",xaxis_rangeslider_visible=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=pd.to_datetime(backtestInterval['Date'].iloc[1:-1]), 
                             y=backtestInterval['backtest_returns'].iloc[1:-1],
                             name='Backtested Strategy Returns (day)',
                             mode='markers+lines',
                            marker={"size":12},
                            marker_color=backtestInterval['traded']))
    fig2.add_trace(go.Scatter(x=backtestInterval['Date'].iloc[:-1], 
                             y=np.exp(backtestInterval['log_returns']), 
                             name='Benchmark returns (day)'))
    fig2.update_layout(title="Backtested Strategy vs Buy and Hold Returns (Daily)",xaxis_rangeslider_visible=True)
    summary = {}
    summary['totalPNL'] = ((tradeBook['exitPrice'] -tradeBook['entryPrice']) * tradeBook['units']).dropna().sum()
    summary['meanPNL'] = ((tradeBook['exitPrice'] -tradeBook['entryPrice']) * tradeBook['units']).dropna().mean()
    summary['sharpe'] = empyrical.sharpe_ratio(backtestInterval.iloc[1:-1]['backtest_returns']-1,risk_free=1.07**(1/365)-1)
    summary['max_drawdown'] = empyrical.max_drawdown(backtestInterval.iloc[1:-1]['backtest_returns']-1)
    summary['volatility'] = backtestInterval.iloc[1:-1]['backtest_returns'].std()
    summary['sortino'] = empyrical.sortino_ratio(backtestInterval.iloc[1:-1]['backtest_returns']-1,required_return=1.07**(1/365)-1)
    summary['winRate'] = (((tradeBook['exitPrice'] -tradeBook['entryPrice']) * tradeBook['units']) > 0).dropna().mean()
    summary['meanTime'] = (pd.to_datetime(tradeBook['exitDate']) - pd.to_datetime(tradeBook['entryDate'])).dropna().mean().days
    summary['no_of_trades'] = tradeBook.shape[0]
    summary['bestTrade'] = ((tradeBook['exitPrice'] -tradeBook['entryPrice']) * tradeBook['units']).max()
    summary['worstTrade'] = ((tradeBook['exitPrice'] -tradeBook['entryPrice']) * tradeBook['units']).min()
    summary['totalReturn'] = backtestInterval['backtest_returns'].iloc[1:-1].cumprod().iloc[-1]
    summary['meanReturn'] = backtestInterval['backtest_returns'].iloc[1:-1].mean()
    summary['alpha'] = backtestInterval['backtest_returns'].iloc[1:-1].cumprod().iloc[-1] - np.exp((backtestInterval['log_returns']).iloc[:-1].cumsum()).iloc[-1]
    summary['wins'] = (((tradeBook['exitPrice'] -tradeBook['entryPrice']) * tradeBook['units']) > 0).dropna().sum()
    summary['completedTrades'] = (((tradeBook['exitPrice'] -tradeBook['entryPrice']) * tradeBook['units'])).dropna().count()
    summary['pVal'] = 1-binom.cdf(summary['wins']-1,summary['completedTrades'], 0.5)
    
    if verbose == True:
        print("BACKTEST SUMMARY")
        print("--------------------")
        if swap == True:
            print("Asset Class: DV01-Size")
            print(f"Number of Trades: {summary['no_of_trades']}")
            print(f"WIN RATE: {100*summary['winRate']:.5f}% (Statistical Significance/ p-value : {summary['pVal']}) ")
            print(f"TOTAL PNL: {100*summary['totalPNL']:.2f} ({100*float(summary['totalReturn']/starting_cap)})")
            print(f"MEAN PNL per trade: {100*summary['meanPNL']:.2f} ({100*float(summary['meanReturn']/starting_cap)})")
            print(f"Maximum Drawdown: {100*summary['max_drawdown']:.5f} %")
            print(f"Volatility: {summary['volatility']:.5f}")
            print(f"Sharpe: {summary['sharpe']:.5f}")
            print(f"Sortino: {summary['sortino']:.5f}")
            print(f"Alpha (excess on benchmark): {summary['alpha']:.5f}")
            print(f"Mean Duration of Trade: {summary['meanTime']}")
            print(f"Best Trade: {100*summary['bestTrade']:.2f} ({100*summary['bestTrade']/starting_cap:.2f})")
            print(f"Worst Trade: {100*summary['worstTrade']:.2f} ({100*summary['bestTrade']/starting_cap:.2f})")
        else:
            print("UNITS: NOTATIONAL")
            print(f"Number of Trades: {summary['no_of_trades']}")
            print(f"WIN RATE: {100*summary['winRate']:.5f}% (Statistical Significance/ p-value : {summary['pVal']}) ")
            print(f"TOTAL PNL: {summary['totalPNL']:.2f} ({float(summary['totalReturn']/starting_cap)})")
            print(f"MEAN PNL per trade: {summary['meanPNL']:.2f} ({float(summary['meanReturn']/starting_cap)})")
            print(f"Maximum Drawdown: {100*summary['max_drawdown']:.5f} %")
            print(f"Volatility: {summary['volatility']:.5f}")
            print(f"Sharpe: {summary['sharpe']:.5f}")
            print(f"Sortino: {summary['sortino']:.5f}")
            print(f"Alpha (excess on benchmark): {summary['alpha']:.5f}")
            print(f"Mean Duration of Trade: {summary['meanTime']}")
            print(f"Best Trade: {summary['bestTrade']:.2f} ({summary['bestTrade']/starting_cap:.2f})")
            print(f"Worst Trade: {summary['worstTrade']:.2f} ({summary['bestTrade']/starting_cap:.2f})")
        print("--------------------\n")
    return fig, fig2, summary




def tradeBookPlotter(ID, tradeBook, data_tree):
    period = data_tree['date_test'].loc[(data_tree['date_test'] >= tradeBook.iloc[ID]['entryDate']) & (data_tree['date_test'] <= tradeBook.iloc[ID]['exitDate'])]
    prices = data_tree['X_test'].loc[period.index]
    fig = go.Figure()
    direction = tradeBook.loc[ID, 'units']
    colors = {0:"green",1:"red"}
    fig.add_trace(go.Scatter(x=period,y=prices['Price'], mode='markers+lines',name='Actual Price',marker_size=12,))
    fig.add_trace(go.Scatter(x=[tradeBook.loc[ID,'entryDate']],
                             y=[tradeBook.loc[ID,'entryPrice']],
                             mode='markers',
                             marker_size=20,
                             marker_symbol='x',
                             marker_color= colors[(direction < 0) * 1],
                            name=f"Entry - {tradeBook.loc[ID,'entryReason']}: {direction:.3f} units"))
    fig.add_trace(go.Scatter(x=[tradeBook.loc[ID,'exitDate']],
                             y=[tradeBook.loc[ID,'exitPrice']],
                             mode='markers',
                             marker_size=20,
                             marker_symbol='x',
                             marker_color = colors[1- (direction < 0) * 1],
                            name=f"Exit: {tradeBook.loc[ID,'exitReason']}"))
    profit = (tradeBook.loc[ID, 'exitPrice'] - tradeBook.loc[ID, 'entryPrice']) * tradeBook.loc[ID,'units']
    fig.update_layout(title=f"Trade ID: {ID} - Profit (Loss): {profit:.2f}")
    return fig