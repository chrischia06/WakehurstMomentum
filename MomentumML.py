import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, date

from statsmodels.tsa.stattools import adfuller
import catboost 
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, roc_auc_score,f1_score, confusion_matrix
from sklearn.gaussian_process import GaussianProcessRegressor,GaussianProcessClassifier
from sklearn.cluster import KMeans

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LSTM, Flatten,Input, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2
import shap
import json

def modelHistoricPredsWriter(preds, asset, ID, path=""):
    temp = pd.json_normalize(preds)
    temp['asset'] = asset
    temp['date'] = pd.to_datetime(datetime.now())
    temp['forecast_var'] = param
    temp['train_end'] = train_end
    temp['train_start'] = train_start
    temp['LAGS'] = LAGS
    temp['hyperParams'] = str(hyperParams)
    temp['trainPerformanceID'] = ID
    try:
        modelHistory = pd.read_csv(f'{path}modelHistory.csv')
        modelHistory = pd.concat([modelHistory,temp],axis=0).reset_index(drop=True).drop_duplicates()
    except:
        modelHistory = temp
    modelHistory.to_csv(f"{path}modelHistory.csv",index=False)
    return modelHistory

def trainTestWrapper(df2, df3, train_end, param, normalizeFeats,hyperParams, LAGS):
    data_tree, data, models, logs = trainWrapper(df2, df3, train_end, param, normalizeFeats,hyperParams, LAGS)
    fig3, fig4,logs2 = diagnosis_test(models, data_tree, data, param)
    logs3 = combineLogs(logs, logs2, hyperParams)
    if param == 'direction':
        logs3 = logs3.sort_values('test_accuracy', ascending=False)
    else:
        logs3 = logs3.sort_values('test_mean_squared_error')
    fig, ax = plt.subplots(figsize=(20,5),ncols=3)
    if param in ['log_returns','IntradayRange','Price','Rate']:
        metrics = ['r2_score','mean_squared_error','mean_absolute_error']
    elif param == 'direction':
        metrics = ['accuracy','f1_score','roc_auc']
    for i,x in enumerate(metrics):
        ax[i] = diagnoseTrainTest(logs3,x, ax[i])

    return data_tree, data, models, fig3, fig4, logs3, ax

def combineLogs(log1, log2, hyperParams):
    temp = pd.concat([log1, log2],axis=1)
    temp.columns = ["train_"+x for x in list(log1.columns)] + ["test_"+x for x in list(log2.columns)]
    temp['hyperParams'] = np.nan
    for x in ['dnn','ensemble']:
        temp.loc[x,'hyperParams'] = json.dumps(hyperParams[x])
    if 'xgbc' in temp.index:
        temp.loc['xgbc','hyperParams'] = json.dumps(hyperParams['xgb'])
        temp.loc['cbc','hyperParams'] = json.dumps(hyperParams['cb'])
    else:
        temp.loc['xgbr','hyperParams'] = json.dumps(hyperParams['xgb'])
        temp.loc['cbr','hyperParams'] = json.dumps(hyperParams['cb'])
    return temp

def trainWrapper(df2, df3, train_end, param, normalizeFeats,hyperParams, LAGS):
    if param =='Rate':
        param = 'Price'
    data_tree = createTrainTest(df2, train_end , param=param)
    data = createTrainTest(df3, train_end, param=param)
    if param == 'direction':
        MODE = 'classification'
    else:
        MODE = 'regression'
    models, logs = autofit(data_tree, data,normalizeFeats, hyperParams, LAGS, mode=MODE)
    return data_tree, data, models, logs




def createTrainTest(df, train_end, param='returns'):
    trainDict = {}
    df2 = df.copy()
    if param == 'Price' or param =='Rate':
        df2['Price_next'] = df2['Price'].shift(-1)
    elif param == 'returns':
        df2['returns_next'] = df2['returns'].shift(-1)
    elif param == 'log_returns':
        df2['log_returns_next'] = df2['log_returns'].shift(-1)
    elif param == 'direction':
        df2['direction_next'] = (df2['log_returns'].shift(-1) > 0) * 1.0
    elif param == 'IntradayRange':
        df2['IntradayRange_next'] = df2['IntradayRange'].shift(-1)
    else:
        print("ERROR: INVALID PARAMETER")
        return trainDict
    X_test = df2.loc[df2['Date'] >= train_end].drop([param+"_next",'Date'],axis=1)
    date_test = df2.loc[df2['Date'] >= train_end,'Date']
    y_test = df2.loc[df2['Date'] >= train_end,param+"_next"]
    
    
    trainDict['X_train'] = df2.loc[df2['Date'] < train_end].drop([param+"_next",'Date'],axis=1)
    trainDict['X_test'] = X_test.iloc[:-1]
    trainDict['y_train'] = df2.loc[df2['Date'] < train_end,param+"_next"]
    trainDict['y_test'] = y_test.iloc[:-1]
    trainDict['last'] = X_test.iloc[-1:]
    trainDict['date_train'] = df2.loc[df2['Date'] < train_end,'Date']
    trainDict['date_test'] = date_test.iloc[:-1]
    trainDict['date_last'] = date_test.iloc[-1:]
    return trainDict 

def ensemble_df(models, data_tree, data, mode):
    if mode == 'regression':
        temp = pd.DataFrame(columns=['cbr','xgbr','linear_reg','dnn'])
        for x in ['cbr','xgbr']:
            temp[x] = models[x].predict(data_tree)
        for x in ['linear_reg','gpr','dnn']:
            temp[x] = models[x].predict(data).reshape(-1)
        
    elif mode == 'classification':
        temp = pd.DataFrame(columns=['cbc','xgbc','logit','dnn'])
        for x in ['cbc','xgbc']:
            temp[x] = models[x].predict_proba(data_tree)[:,1]
        for x in ['logit','gpc']:
            temp[x] = models[x].predict_proba(data)[:,1]
        temp['dnn'] = models['dnn'].predict(data).reshape(-1)
    return temp

feats2 = ['log_returns','IntradayRange']
def autofit(data_tree, data, normalizeFeats, hyperParams, LAGS, mode='regression'):
    models = {}

    if len(normalizeFeats) > 0:
        models['scaler'] = StandardScaler()
        data['X_train'][normalizeFeats] = models['scaler'].fit_transform(data['X_train'][normalizeFeats])
        data['X_test'][normalizeFeats] = models['scaler'].transform(data['X_test'][normalizeFeats])
        data['last'][normalizeFeats] = models['scaler'].transform(data['last'][normalizeFeats])
    
    n_clusters = hyperParams['kmeans']['n_clusters']
    models['kmeans'] = KMeans(n_clusters=n_clusters).fit(data['X_train'][feats2])
    train_clusters, test_clusters, last_cluster = models['kmeans'].predict(data['X_train'][feats2]), models['kmeans'].predict(data['X_test'][feats2]), models['kmeans'].predict(data['last'][feats2])
    data_tree['X_train']['cluster'], data_tree['X_test']['cluster'], data_tree['last']['cluster'] =  train_clusters, test_clusters, last_cluster
    data['X_train']['cluster'], data['X_test']['cluster'], data['last']['cluster'] =  train_clusters, test_clusters, last_cluster

    pool = Pool(data_tree['X_train'], data_tree['y_train'], cat_features=['day','month','weekday','cluster'])

    log_dir = "logs/swaps/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    inputs = Input(data['X_train'].shape[1])

    if mode=='regression':
        models['cbr'] = CatBoostRegressor(n_estimators=hyperParams['cb']['n_estimators'],
                                max_depth=hyperParams['cb']['max_depth'],
                                learning_rate=hyperParams['cb']['learning_rate'],
                                verbose=0).fit(pool)
        models['xgbr'] = XGBRegressor(n_estimators=hyperParams['xgb']['n_estimators'], 
                                        max_depth=hyperParams['xgb']['max_depth'],
                                        learning_rate=hyperParams['xgb']['learning_rate'],objective='reg:squarederror').fit(data_tree['X_train'], data_tree['y_train'])
        models['linear_reg'] = LinearRegression().fit(data['X_train'], data['y_train'])
        models['gpr'] = GaussianProcessRegressor().fit(data['X_train'], data['y_train'])
        models['ensemble'] = CatBoostRegressor(n_estimators=hyperParams['ensemble']['n_estimators'],
                                max_depth=hyperParams['ensemble']['max_depth'],
                                learning_rate=hyperParams['ensemble']['learning_rate'],
                                verbose=0).fit(pool)

        #neural network
        l1 = Dense(hyperParams["dnn"]['hidden_units'],activation='relu', kernel_regularizer=l2(0.01))(inputs)
        outputs = Dense(1,activation='linear')(l1)
        model = Model(inputs,outputs)
        model.compile(loss='mean_squared_error', optimizer='Adam')
        models['dnn'] = model

        models['dnn'].fit(data['X_train'], 
          data['y_train'],
          epochs=30,
          batch_size=16,
          shuffle=False,
          validation_split=0.2,
          verbose=0,
         callbacks=[EarlyStopping(patience=20),tensorboard_callback])

        #ensemble
        data_tree['ensemble_train'] = ensemble_df(models, data_tree['X_train'], data['X_train'], mode)
        data_tree['ensemble_test'] = ensemble_df(models, data_tree['X_test'], data['X_test'], mode)
        data_tree['ensemble_last'] = ensemble_df(models, data_tree['last'], data['last'], mode)

        pool2 = Pool(data_tree['ensemble_train'], data_tree['y_train'])
        models['ensemble'].fit(pool2)
        
        
    elif mode =='classification':
        models['cbc'] = CatBoostClassifier(n_estimators=hyperParams['cb']['n_estimators'],
                                max_depth=hyperParams['cb']['max_depth'],
                                learning_rate=hyperParams['cb']['learning_rate'],
                                verbose=0).fit(pool)
        models['xgbc'] = XGBClassifier(n_estimators=hyperParams['xgb']['n_estimators'], 
                                        max_depth=hyperParams['xgb']['max_depth'],
                                        learning_rate=hyperParams['xgb']['learning_rate']).fit(data_tree['X_train'], data_tree['y_train'])
        models['logit'] = LogisticRegression().fit(data['X_train'], data['y_train'])
        models['gpc'] = GaussianProcessClassifier().fit(data['X_train'], data['y_train'])
        models['ensemble'] = CatBoostClassifier(n_estimators=hyperParams['ensemble']['n_estimators'],
                                max_depth=hyperParams['ensemble']['max_depth'],
                                learning_rate=hyperParams['ensemble']['learning_rate'],
                                verbose=0).fit(pool)


        l1 = Dense(hyperParams["dnn"]['hidden_units'],activation='relu')(inputs)
        outputs = Dense(1,activation='sigmoid')(l1)
        model = Model(inputs,outputs)
        model.compile(loss='binary_crossentropy', optimizer='Adam')
        models['dnn'] = model

        models['dnn'].fit(data['X_train'], 
          data['y_train'],
          epochs=40,
          batch_size=16,
          shuffle=False,
          validation_split=0.2,
          verbose=0,
         callbacks=[EarlyStopping(patience=20),tensorboard_callback])

        #ensemble
        data_tree['ensemble_train'] = ensemble_df(models,data_tree['X_train'],data['X_train'], mode)
        data_tree['ensemble_test'] = ensemble_df(models,data_tree['X_test'],data['X_test'], mode)
        data_tree['ensemble_last'] = ensemble_df(models,data_tree['last'],data['last'], mode)

        pool2 = Pool(data_tree['ensemble_train'], data_tree['y_train'])
        models['ensemble'].fit(pool2)
        

    logs = diagnosis_train(models, data_tree, data, mode)
    return models, logs


def regressionDiagnosis(y_actual, y_pred, verbose=False):
    res = {}
    res['mean_squared_error'] = mean_squared_error(y_actual, y_pred)
    res['mean_absolute_error'] = mean_absolute_error(y_actual, y_pred)
    res['r2_score'] = r2_score(y_actual, y_pred)
    if verbose == True:
        print(f"Mean Squared Error: {res['mean_squared_error']}")
        print(f"Mean Absolute Error: {res['mean_absolute_error']}")
        print(f"R^2: {res['r2_score']}")
        print("--------------------")
    return res


def classificationDiagnosis(y_actual, y_pred, y_proba, verbose=False):
    res = {}
    res['accuracy'] = accuracy_score(y_actual, y_pred)
    res['f1_score'] = f1_score(y_actual, y_pred)
    if verbose == True:
        print(f"Accuracy: {res['accuracy']}")
        print(f"F1 Score: {res['f1_score']}")
    try:
        res['roc_auc'] = roc_auc_score(y_actual, y_proba)
        if verbose == True:
            print(f"ROC AUC Score: {res['roc_auc']}")
    except:
        res['roc_auc'] = np.nan
        print("Only 1 Target in dataset")
    if verbose == True:    
        print("--------------------")
    return res

def featImportances(models,data_tree, data, param):
    featImportances  = {}
    fig, ax = plt.subplots(figsize=(10,40),nrows=4)

    if param == 'direction':
        mode = 'classification'
    else:
        mode = 'regression'

    if mode == 'regression':
        featImportances['linear_reg'] = dict(sorted(zip(list(data['X_train'].columns), models['linear_reg'].coef_), key = lambda k : abs(k[1])))
        featImportances['cbr'] = dict(sorted(zip(list(data_tree['X_train'].columns), models['cbr'].get_feature_importance()), key = lambda k : k[1]))
        featImportances['ensemble'] = dict(sorted(zip(list(data_tree['ensemble_train'].columns), models['ensemble'].get_feature_importance()), key = lambda k : k[1]))
        ax[0].barh(list(featImportances['linear_reg'].keys())[:20], list(featImportances['linear_reg'].values())[:20])
        ax[0].set_title("Linear Regression Coefficients")
        ax[1].barh(list(featImportances['cbr'].keys())[:20], list(featImportances['cbr'].values())[:20])
        ax[2].barh(list(featImportances['ensemble'].keys())[:20], list(featImportances['ensemble'].values())[:20])
        xgb.plot_importance(models['xgbr'], ax[3], max_num_features=20,title='XGBoost Feature Importance')
    elif mode == 'classification':
        featImportances['logit'] = dict(sorted(zip(list(data['X_train'].columns), models['logit'].coef_[0]), key = lambda k : abs(k[1])))
        featImportances['cbc'] = dict(sorted(zip(list(data_tree['X_train'].columns), models['cbc'].get_feature_importance()), key = lambda k : k[1]))
        featImportances['ensemble'] = dict(sorted(zip(list(data_tree['ensemble_train'].columns), models['ensemble'].get_feature_importance()), key = lambda k : k[1]))
        ax[0].barh(list(featImportances['logit'].keys())[:20], list(featImportances['logit'].values())[:20])
        ax[0].set_title("Logistic Regression coefficients")
        ax[1].barh(list(featImportances['cbc'].keys())[:20], list(featImportances['cbc'].values())[:20])
        ax[2].barh(list(featImportances['ensemble'].keys())[:20], list(featImportances['ensemble'].values())[:20])
        xgb.plot_importance(models['xgbc'], ax[3], max_num_features=20, title='XGBoost Feature Importance')
    ax[1].set_title("Catboost Feature Importance")
    ax[2].set_title("Ensemble Feature Importance")
    return ax



def diagnosis_train(models, data_tree, data, mode='regression'):
    print("MODEL EVALUATION - TRAINING SET")
    print("--------------------")
    logs = {}
    
    if mode == 'regression':
        for x in ['cbr','xgbr']:
            print(f"Model: {x}")
            logs[x] = regressionDiagnosis(data_tree['y_train'], models[x].predict(data_tree['X_train']))
        for x in ['linear_reg','dnn','gpr']:
            print(f"Model: {x}")
            logs[x] = regressionDiagnosis(data['y_train'], models[x].predict(data['X_train']).reshape(-1))            
        for x in ['ensemble']:
            print(f"Model: {x}")
            logs[x] = regressionDiagnosis(data_tree['y_train'], models[x].predict(data_tree['ensemble_train']).reshape(-1))        

        logs2 = pd.DataFrame(columns=['mean_squared_error','mean_absolute_error','r2_score'])
        for x in logs:
            logs2.loc[x] = logs[x]
        print("--------------------\n")
        return logs2

    elif mode == 'classification':
        for x in ['cbc','xgbc']:
            print(f"Model: {x}")
            logs[x] = classificationDiagnosis(data_tree['y_train'], 
                                            models[x].predict(data_tree['X_train']), 
                                            models[x].predict_proba(data_tree['X_train'])[:,1])           
        for x in ['logit','gpc']:
            print(f"Model: {x}")
            logs[x] = classificationDiagnosis(data['y_train'], models[x].predict(data['X_train']), models[x].predict_proba(data['X_train'])[:,1])
         
        for x in ['dnn']:
            print(f"Model: {x}")
            logs[x] = classificationDiagnosis(data['y_train'], np.round(models[x].predict(data['X_train']).reshape(-1)), models[x].predict(data['X_train']))
        for x in ['ensemble']:
            print(f"Model: {x}")
            logs[x] = classificationDiagnosis(data_tree['y_train'], 
                                            models[x].predict(data_tree['ensemble_train']), 
                                            models[x].predict_proba(data_tree['ensemble_train'])[:,1])
            
        
        logs2 = pd.DataFrame(columns=['accuracy','f1_score','roc_auc'])
        for x in logs:
            logs2.loc[x] = logs[x]
        print("--------------------\n")
        return logs2






def diagnosis_test(models, data_tree, data, param):
    print("MODEL EVALUATION - TEST SET")
    print("--------------------")
    if param == 'direction':
        mode = 'classification'
    else:
        mode = 'regression'
    logs = {}
    modelKeys = ['cbc', 'xgbc', 'logit', 'gpc', 'ensemble', 'dnn']
    if mode =='regression':
        fig = go.Figure()
        fig2 = go.Figure()
        for x in ['cbr','xgbr']:
            print(f"Model: {x}")
            logs[x] = regressionDiagnosis(data_tree['y_test'], models[x].predict(data_tree['X_test']))
            fig.add_trace(go.Scatter(x=data_tree['date_test'],y=models[x].predict(data_tree['X_test']),name=f"preds-{x}"))
            fig2.add_trace(go.Scatter(x=data_tree['date_test'],y=data_tree['y_test']-models[x].predict(data_tree['X_test']),name=f"resids-{x}"))
        for x in ['linear_reg', 'dnn','gpr']:
            print(f"Model: {x}")
            logs[x] = regressionDiagnosis(data['y_test'], models[x].predict(data['X_test']).reshape(-1))
            fig.add_trace(go.Scatter(x=data['date_test'],y=models[x].predict(data['X_test']).reshape(-1),name=f"preds-{x}"))
            fig2.add_trace(go.Scatter(x=data['date_test'],y=data['y_test']-models[x].predict(data['X_test']).reshape(-1),name=f"resids-{x}"))

        for x in ['ensemble']:
            print(f"Model: {x}")
            logs[x] = regressionDiagnosis(data_tree['y_test'], models[x].predict(data_tree['ensemble_test']))
            fig.add_trace(go.Scatter(x=data_tree['date_test'],y=models[x].predict(data_tree['ensemble_test']),name=f"preds-{x}"))
            fig2.add_trace(go.Scatter(x=data_tree['date_test'],y=data_tree['y_test']-models[x].predict(data_tree['ensemble_test']),name=f"resids-{x}"))

        fig.add_trace(go.Scatter(x=data_tree['date_test'],y=data_tree['y_test'],name='actual', mode='markers+lines',marker={"size":12}))
        fig.update_layout(title='Predictions')
        fig2.update_layout(title='Residuals')
        logs2 = pd.DataFrame(columns=['mean_squared_error','mean_absolute_error','r2_score'])
        for x in logs:
            logs2.loc[x] = logs[x]
        return fig2, fig , logs2
    elif mode =='classification':
        fig, ax = plt.subplots(figsize=(5*len(modelKeys),5), ncols=len(modelKeys))
        fig2 = go.Figure()
        for x in enumerate(modelKeys):
            if x[1] in ['cbc','xgbc']:
                print(f"Model: {x[1]}")
                logs[x[1]] = classificationDiagnosis(data_tree['y_test'], models[x[1]].predict(data_tree['X_test']), models[x[1]].predict_proba(data_tree['X_test'])[:,1])
                ax[x[0]].set_title(f"Confusion Matrix - {x[1]}")
                sns.heatmap(confusion_matrix(data_tree['y_test'], models[x[1]].predict(data_tree['X_test'])), ax =ax[x[0]], annot =True)
                fig2.add_trace(go.Scatter(x=data_tree['date_test'],y=models[x[1]].predict_proba(data_tree['X_test'])[:,1],name=f"Predictions-{x[1]}"))
            if x[1] in ['logit','gpc']:
                print(f"Model: {x[1]}")
                logs[x[1]] = classificationDiagnosis(data_tree['y_test'], models[x[1]].predict(data['X_test']), models[x[1]].predict_proba(data['X_test'])[:,1])
                ax[x[0]].set_title(f"Confusion Matrix - {x[1]}")
                sns.heatmap(confusion_matrix(data['y_test'], models[x[1]].predict(data['X_test'])), ax =ax[x[0]], annot =True)
                fig2.add_trace(go.Scatter(x=data['date_test'],y=models[x[1]].predict_proba(data['X_test'])[:,1],name=f"Predictions-{x[1]}"))
            if x[1] in ['dnn']:
                print(f"Model: {x[1]}")
                logs[x[1]] = classificationDiagnosis(data['y_test'], np.round(models[x[1]].predict(data['X_test']).reshape(-1)), models[x[1]].predict(data['X_test']))
                ax[x[0]].set_title(f"Confusion Matrix - {x[1]}")
                sns.heatmap(confusion_matrix(data['y_test'], np.round(models[x[1]].predict(data['X_test']).reshape(-1))), ax =ax[x[0]], annot =True)
                fig2.add_trace(go.Scatter(x=data['date_test'],y=models[x[1]].predict(data['X_test']).reshape(-1),name=f"Predictions-{x[1]}"))
            if x[1] in ['ensemble']:
                print(f"Model: {x[1]}")
                logs[x[1]] = classificationDiagnosis(data_tree['y_test'], 
                                            models[x[1]].predict(data_tree['ensemble_test']), 
                                            models[x[1]].predict_proba(data_tree['ensemble_test'])[:,1])
                ax[x[0]].set_title(f"Confusion Matrix - {x[1]}")
                sns.heatmap(confusion_matrix(data_tree['y_test'], models[x[1]].predict(data_tree['ensemble_test'])), ax =ax[x[0]], annot =True)
                fig2.add_trace(go.Scatter(x=data_tree['date_test'],y=models[x[1]].predict_proba(data_tree['ensemble_test'])[:,1],name=f"Predictions-{x[1]}"))



        fig2.add_trace(go.Scatter(x=data_tree['date_test'],y=data_tree['y_test'], mode='markers',name="Actual Direction",
                           marker_color=1-data_tree['y_test'],
                           marker_symbol=data_tree['y_test']*3,
                           marker_line_width=1,
                           marker={"size":12,"colorscale":"Bluered"}))
        fig2.update_layout(title='Model Predictions',xaxis_rangeslider_visible=True)

        logs2 = pd.DataFrame(columns=['accuracy','f1_score','roc_auc'])
        for x in logs:
            logs2.loc[x] = logs[x]
    return fig2, ax, logs2
    

def diagnoseTrainTest(logs, metric, ax):
    if metric in ['f1_score','roc_auc','accuracy']:
        ax.plot(np.arange(0.5,1,0.05), np.arange(0.5,1,0.05))
    ax.scatter(logs[f'train_{metric}'],logs[f'test_{metric}'])
    ax.set_title(f'Train vs Test {metric}')
    ax.set_xlabel(f'Train {metric}')
    ax.set_ylabel(f'Train {metric}')
    for i, model in enumerate(logs.index.values):
        ax.annotate(model, (logs[f'train_{metric}'].values[i]*0.95,logs[f'test_{metric}'].values[i]))
        

                

def recommendTrade(data_tree, data, models, param, swap=False):
    print("LAST DATE:", str(data_tree['date_last'].values[0])[:10])
    print("----------")

    preds = {}
    if param == 'direction':
        if swap == True:
            print("FORECAST PROBABILITY OF INCREASE IN RATE (0 means rate very likely to decrease, 1 very likely to increase) ")
        else:
            print("FORECAST PROBABILITY OF INCREASE IN PRICE (0 means price very likely to decrease, 1 very likely to increase) ")
        print("----------")
        preds["xgbc"] = models['xgbc'].predict_proba(data_tree['last'])[:,1][0]
        preds["cbc"] = models['cbc'].predict_proba(data_tree['last'])[:,1][0]
        preds["logit"] = models['logit'].predict_proba(data['last'])[:,1][0]
        preds["gpc"] = models['gpc'].predict_proba(data['last'])[:,1][0]
        preds["dnn"] = models['dnn'].predict(data['last'])[0][0]
        preds["ensemble"] = models['ensemble'].predict_proba(data_tree['ensemble_last'])[:,1][0]
        print(f'XGBoost Classifier: {preds["xgbc"]:.5f}')
        print(f'CatBoost Classifier {preds["cbc"]:.5f}')
        print(f'Logit: {preds["logit"]:.5f}')
        print(f'Deep Neural Network: {preds["dnn"]:.5f}')
        print(f'Gaussian Process: {preds["gpc"]:.5f}')
        print(f'Ensemble: {preds["ensemble"]:.5f}')

    else:
        if param == 'returns':
            print("FORECAST PERCENTAGE CHANGE IN ", end="")
        elif param == 'log_returns':
            print("FORECAST LOG-CHANGE IN ", end="")
        elif param == 'Price':
            print("FORECAST ", end="")
        if swap == True:
            print("RATE")
        else:
            print("PRICE")
        print("----------")
        preds['xgbr'] = models['xgbr'].predict(data_tree['last'])[0]
        preds['cbr'] = models['cbr'].predict(data_tree['last'])[0]
        preds['linear_reg'] = models['linear_reg'].predict(data['last'])[0]
        preds['gpr'] = models['gpr'].predict(data['last'])[0]
        preds['dnn'] = models['dnn'].predict(data['last'])[0][0]
        preds['ensemble'] = models['ensemble'].predict(data_tree['ensemble_last'])[0]
        print(f"XGBoost: {preds['xgbr']:.5f}")
        print(f"Catboost: {preds['cbr']:.5f}")
        print(f"Linear Regression: {preds['linear_reg']:.5f}")
        print(f"DNN: {preds['gpr']:.5f}")
        print(f"Gaussian Process: {preds['dnn']:.5f}")
        print(f"Ensemble: {preds['ensemble']:.5f}")
        print(f"Price: {np.exp(data_tree['last']['Price'].values[0]):.5f}")
    return preds
        

def shapPlotter(models, model, data_tree, data):
    if model in set({'xgbc','xgbr'}):
        explainer = shap.TreeExplainer(models[model])
        shap_values = explainer.shap_values(data_tree['last'])
        return shap.force_plot(explainer.expected_value,shap_values, data_tree['last'])
    elif model == 'ensemble':
        explainer = shap.TreeExplainer(models[model])
        shap_values = explainer.shap_values(data_tree['ensemble_last'])
        return shap.force_plot(explainer.expected_value,shap_values, data_tree['ensemble_last'])
    elif model in set({'logit','gpc'}):
        explainer = shap.KernelExplainer(models['logit'].predict_proba, data['last'], link="logit")
        shap_values = explainer.shap_values(data['last'])
        return shap.force_plot(explainer.expected_value[1],shap_values[0], data['last'])
#     elif model == 'dnn':
#         explainer = shap.DeepExplainer(models['dnn'], data['last'].values)
#         shap_values = explainer.shap_values(data['last'])
#         return shap.force_plot(explainer.expected_value[1],shap_values[0], data['last'])
    elif model in set({'cbc','cbr'}):
        explainer = shap.TreeExplainer(models[model])
        shap_values = models[model].get_feature_importance(data=Pool(data_tree['last'],cat_features=['day','month','weekday','cluster']),type='ShapValues')
        return shap.force_plot(explainer.expected_value,shap_values[0][:-1], data_tree['last'])
        
