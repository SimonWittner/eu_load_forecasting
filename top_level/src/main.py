import argparse
import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet, set_log_level, save
import mlflow
import os
import time
from azureml.core import Run
import logging
#from memory_profiler import profile

  
from error_metrics import _calc_mae, _calc_mse, _calc_rmse, _calc_nrmse, _calc_mape, _calc_mase, _calc_msse, _seas_naive_fcst, _calc_metrics

#@profile
def main():
    """Main function of the script."""

    # input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")

    #output arguments
    parser.add_argument("--results", type=str, help="path to results")

    args = parser.parse_args()

    temp_name = 'temp'
 
    ##################
    #<load data>
    ##################
    start = time.time()

    print(f'starting time: {start}')

    df = pd.read_csv(args.data, dtype={'country': str, 'y': 'float32', temp_name: 'float32'}, parse_dates=['ds'])

    # rename columns
    df = df.rename(columns={'country': 'ID'})

    # only use the first 50 IDs
    IDs = df['ID'].unique()

    df = df.copy() # add seperate columns for mon_winter, tue_winter, etc. and mon_summer, tue_summer, etc.
    df['winter'] = np.where(df['ds'].dt.month.isin([10,11,12,1,2,3,]), 1, 0)
    df['summer'] = np.where(df['ds'].dt.month.isin([4,5,6,7,8,9]), 1, 0)

    df = df[['ds', 'y', 'ID', temp_name, 'humidity', 'precipitation', 'cloud', 'wind', 'winter', 'summer']]

    # normalize Temp
    df[temp_name] = (df[temp_name] - 18.33) / 10

    ##################
    #<train the model>
    ##################
    tuned_params = {
        'n_lags':24*15,
        'newer_samples_weight': 2,
        'n_changepoints': 0,
        'yearly_seasonality': 10,
        'weekly_seasonality': True,
        'daily_seasonality': True,
        'batch_size': 128,
        'ar_layers': [32, 64, 32, 16],
        'lagged_reg_layers': [32, 32],
        # not tuned
        'n_forecasts': 33,
        'learning_rate': 0.001,
        'epochs': 30, 
        #'epochs': 3, 
        #'trend_global_local': 'local',  #!! comment out with multiple IDs
        'season_global_local': 'local',
        'drop_missing': True,
        'normalize': 'standardize',
    }

    trainer_configs = {
        'accelerator': 'gpu',
    } 

    # AENDERUNG SIMON: Parameter for quantile regression
    confidence_lv = 0.98
    quantile_list = [round(((1 - confidence_lv) / 2), 2), round((confidence_lv + (1 - confidence_lv) / 2), 2)]

    mlflow.pytorch.autolog(log_every_n_epoch=10, log_models=False, log_datasets=False)

    # AENDERUNG SIMON: build and train the model with these hyper-parameters
    m = NeuralProphet(**tuned_params, **trainer_configs, quantiles=quantile_list)

    # split data into train, val, test
    #OLD df_train, df_test = m.split_df(df, valid_p=0.2)
    df_train = df[df['ds'] < '2014-01-01']
    df_test = df[df['ds'] >= '2014-01-01']

    # print start and end dates of df_test
    print(df_test.groupby('ID')['ds'].agg(['min', 'max']))
    print(f'Number of IDs in train: {df_train["ID"].nunique()}, test: {df_test["ID"].nunique()}')


    set_log_level("INFO")

    # add lagged regressor
    m.add_lagged_regressor(names=temp_name, n_lags=33, normalize='standardize')
    m.add_lagged_regressor(names='humidity', n_lags=33, normalize='standardize')
    m.add_lagged_regressor(names='precipitation', n_lags=33, normalize='standardize')
    m.add_lagged_regressor(names='cloud', n_lags=33, normalize='standardize')
    m.add_lagged_regressor(names='wind', n_lags=33, normalize='standardize')

    # add conditional seasonality
    m.add_seasonality(name='winter', period=1, fourier_order=6, condition_name='winter')
    m.add_seasonality(name='summer', period=1, fourier_order=6, condition_name='summer')

    # add holidays
    #m.add_country_holidays(country_name='US', lower_window=-1, upper_window=1)

    print('start training')
    with mlflow.start_run() as run:
        metrics = m.fit(df = df_train, freq="H", num_workers=4, early_stopping=True)
        forecasts_test = m.predict(df_test)

    end = time.time()


    ##################
    #<evaluate the model>
    ##################

    df_metrics_all_countrys_test, df_metrics_all_countrys_averaged_test = _calc_metrics(forecasts=forecasts_test, start_forecast=10, n_forecasts=m.n_forecasts, metric_names=['RMSE', 'MAE', 'MAPE', 'MASE', 'MSSE'])
    
    ##########################
    #<save forecasts and model>
    ##########################

    # log parameters
    run_id = Run.get_context(allow_offline=True).id
    
    with mlflow.start_run(run_id):
        # log metrics
        mlflow.log_metric('duration', end-start)

        mlflow.log_metric("MAE_train", list(metrics['MAE'])[-1])
        mlflow.log_metric("RMSE_train", list(metrics['RMSE'])[-1])
        mlflow.log_metric("Loss_train", list(metrics['Loss'])[-1])

        mlflow.log_metric("RMSE_test_final", df_metrics_all_countrys_averaged_test['RMSE'][0])
        mlflow.log_metric("MAE_test_final", df_metrics_all_countrys_averaged_test['MAE'][0])
        mlflow.log_metric("MAPE_test_final", df_metrics_all_countrys_averaged_test['MAPE'][0])
        mlflow.log_metric("MASE_test_final", df_metrics_all_countrys_averaged_test['MASE'][0])
        mlflow.log_metric("MSSE_test_final", df_metrics_all_countrys_averaged_test['MSSE'][0])

    # save forecasts
    forecasts_test.to_csv(os.path.join(args.results, "forecasts_forecast_test.csv"), index=False)

    # save model
    save(m, os.path.join(args.results, f"model.np"))

    # save metrics
    metrics.to_csv(os.path.join(args.results, "metrics_metrics_train.csv"), index=False)
    df_metrics_all_countrys_test.to_csv(os.path.join(args.results, "metrics_metrics_test.csv"), index=False)

    

if __name__ == "__main__":
    main()
