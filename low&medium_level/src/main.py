import argparse
import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet, set_log_level, save
import mlflow
import os
import time
from azureml.core import Run
import logging
from error_metrics import _calc_mae, _calc_mse, _calc_rmse, _calc_nrmse, _calc_mape, _calc_mase, _calc_msse, _seas_naive_fcst, _calc_metrics

def main():
    """Main function of the script including conditional seasonality."""

    ## Input
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
        
    ## Output
    parser.add_argument("--results", type=str, help="path to results")
    args = parser.parse_args()

    ################## 
    #<load data>
    ##################
    ## Preparation
    start = time.time()
    logging.info(f'starting time: {start}')
    print(f'starting time: {start}')

    df = pd.read_csv(args.data, parse_dates=['ds'])

    country_list = df['country'].unique().tolist()

    df['ID'] = df['ID'].astype(str)
    df['ds'] = pd.to_datetime(df['ds'])
    df['t2m'] = pd.to_numeric(df['t2m'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df['y'] = df['y'].abs()

    df = df.copy() # add seperate columns for mon_winter, tue_winter, etc. and mon_summer, tue_summer, etc.
    df['winter'] = np.where(df['ds'].dt.month.isin([10,11,12,1,2,3,]), 1, 0)
    df['summer'] = np.where(df['ds'].dt.month.isin([4,5,6,7,8,9]), 1, 0)

    df['winter'] = pd.to_numeric(df['winter'], errors='coerce')
    df['summer'] = pd.to_numeric(df['summer'], errors='coerce')

    df['winter'] = df['winter'].astype(bool)
    df['summer'] = df['summer'].astype(bool)
    df = df[['ds', 'ID', 't2m', 'y', 'winter', 'summer']]

    ## Normalize t2m - Kelvin!
    df['t2m'] = (df['t2m'] - 291) / 283

    if df['ID'].nunique() == 1:
        use_one_ID = True
    else:
        use_one_ID = False

    print(df.head(5))
    print(df.tail(5))

    ################## 
    #<train the model>
    ##################
    ## Hyperparameters
    tuned_params = {
        'n_lags':24*15,
        'newer_samples_weight': 2.0,
        'n_changepoints': 0, 
        'yearly_seasonality': 10,
        'weekly_seasonality': True, 
        'daily_seasonality': False, 
        'batch_size': 128,
        #'ar_layers': [32, 64, 32, 16],
        #'lagged_reg_layers': [32, 32],
        'ar_layers': [32, 64, 32],
        'lagged_reg_layers': [32, 64, 32],
        # not tuned
        'n_forecasts': 33,
        'learning_rate': 0.001,
        'epochs': 30, 
        'trend_global_local': 'local' if not use_one_ID else 'global',
        'season_global_local': 'local' if not use_one_ID else 'global',
        'drop_missing': True,
        'normalize': 'standardize',
    }

    trainer_configs = {
        'accelerator': 'gpu',
    }

    ## Uncertainty quantification
    confidence_lv = 0.98
    quantile_list = [round(((1 - confidence_lv) / 2), 2), round((confidence_lv + (1 - confidence_lv) / 2), 2)]
    mlflow.pytorch.autolog(log_every_n_epoch=1, log_models=False, log_datasets=False)

    ## Model
    m = NeuralProphet(**tuned_params, **trainer_configs, quantiles=quantile_list)

    ## Data split
    df_train = df[df['ds'] < '2014-01-01']
    df_test = df[df['ds'] >= '2014-01-01']
    
    print('start and end dates of df_train')
    print(df_train.groupby('ID')['ds'].agg(['min', 'max']))
    print('start and end dates of df_test')
    print(df_test.groupby('ID')['ds'].agg(['min', 'max']))
    print(f'Number of IDs in train: {df_train["ID"].nunique()}, test: {df_test["ID"].nunique()}')

    ## Lagged regressor
    m.add_lagged_regressor(names='t2m', n_lags=33, normalize='standardize')

    ## Conditional seasonality
    m.add_seasonality(name='winter', period=1, fourier_order=6, condition_name='winter')
    m.add_seasonality(name='summer', period=1, fourier_order=6, condition_name='summer')

    ## Holidays
    country_list = ['BLG' if x == 'BGR' else x for x in country_list]
    country_list = ['PRT' if x == 'POR' else x for x in country_list]
    print('German holidays for all countries')
    m.add_country_holidays(country_name='DEU', lower_window=-1, upper_window=1)
    
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
    run_id = Run.get_context(allow_offline=True).id
    
    with mlflow.start_run(run_id):
        mlflow.log_metric('duration', end-start)
        mlflow.log_metric("MAE_train", list(metrics['MAE'])[-1])
        mlflow.log_metric("RMSE_train", list(metrics['RMSE'])[-1])
        mlflow.log_metric("Loss_train", list(metrics['Loss'])[-1])
        mlflow.log_metric("RMSE_test_final", df_metrics_all_countrys_averaged_test['RMSE'][0])
        mlflow.log_metric("MAE_test_final", df_metrics_all_countrys_averaged_test['MAE'][0])
        mlflow.log_metric("MAPE_test_final", df_metrics_all_countrys_averaged_test['MAPE'][0])
        mlflow.log_metric("MASE_test_final", df_metrics_all_countrys_averaged_test['MASE'][0])
        mlflow.log_metric("MSSE_test_final", df_metrics_all_countrys_averaged_test['MSSE'][0])

    # Save forecasts, models and metrics
    forecasts_test.to_csv(os.path.join(args.results, "forecasts_forecast_test.csv"), index=False)
    save(m, os.path.join(args.results, f"model.np"))
    metrics.to_csv(os.path.join(args.results, "metrics_metrics_train.csv"), index=False)
    df_metrics_all_countrys_test.to_csv(os.path.join(args.results, "metrics_metrics_test.csv"), index=False) 

if __name__ == "__main__":
    main()