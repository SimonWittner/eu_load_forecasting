import argparse
import pandas as pd
import numpy as np
import os
import time

import mlflow
from azureml.core import Run

from darts.dataprocessing.transformers import Scaler
from darts.utils.missing_values import fill_missing_values
from darts.timeseries import TimeSeries
from darts.models import XGBModel


from error_metrics import _calc_mae, _calc_mse, _calc_rmse, _calc_nrmse, _calc_mape, _calc_mase, _calc_msse, _seas_naive_fcst, _calc_metrics



def main():
    """Main function of the script."""

   # Input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to input data")

    # Hyperparameters
    parser.add_argument("--learning_rate", type=float, help="Step size shrinkage used in model update")
    parser.add_argument("--subsample", type=float, help="Subsample ratio of the training instances")
    parser.add_argument("--max_leaves", type=int, help="Maximum number of nodes to be added")
    parser.add_argument("--max_depth", type=int, help="Maximum depth of a tree")
    parser.add_argument("--gamma", type=float, help="Minimum loss reduction required for further partition")
    parser.add_argument("--colsample_bytree", type=float, help="Subsample ratio of features/columns")
    parser.add_argument("--min_child_weight", type=float, help="Minimum weights of the instances required in a leaf")
     
    #output arguments
    parser.add_argument("--results", type=str, help="path to results")

    args = parser.parse_args()

    ##################
    #<load data>
    ##################
    start = time.time()

    # log starting time
    print(f'starting time: {start}')

    df = pd.read_csv(args.data, parse_dates=['ds'])

    df['ds'] = pd.to_datetime(df['ds'])
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.rename(columns={'country': 'ID'})
    df['temp'] = pd.to_numeric(df['temp'], errors='coerce')

    df = df[['ds', 'ID', 'temp', 'y']]


    ################## 
    #<train the model>
    ##################

    # generate list of ts
    ts_list = []
    ts_train_list = []
    ts_val_list = []
    past_cov_list = []
    past_cov_train_list = []
    past_cov_val_list = []
    IDs = df['ID'].unique()

    for id in IDs:
        df_ID = df[df['ID'] == id]
        ts = TimeSeries.from_dataframe(df_ID, time_col='ds', value_cols=['y'])
        ts = fill_missing_values(ts)
        ts_train_val, ts_test = ts.split_before(pd.Timestamp('2014-01-01')) # TODO: ADAPT! 9 months from Jan 2021 to Sep 2021
        ts_train, ts_val = ts_train_val.split_before(pd.Timestamp('2014-09-01')) # 9 months from April 2020 to Dec 2021

        past_cov = TimeSeries.from_dataframe(df_ID, time_col='ds', value_cols=['temp'])
        past_cov_train_val, past_cov_test = past_cov.split_before(pd.Timestamp('2014-01-01'))
        past_cov_train, past_cov_val = past_cov_train_val.split_before(pd.Timestamp('2014-09-01'))

        ts_list.append(ts)
        ts_train_list.append(ts_train)
        ts_val_list.append(ts_val)
        past_cov_list.append(past_cov)
        past_cov_train_list.append(past_cov_train)
        past_cov_val_list.append(past_cov_val)

    print('Starting Scaling')
    scaler = Scaler()
    ts_list_scaled = scaler.fit_transform(ts_list)
    ts_train_list_scaled = scaler.transform(ts_train_list)
    ts_val_list_scaled = scaler.transform(ts_val_list)

    tuned_params = {
        'learning_rate': args.learning_rate,
        'subsample': args.subsample,
        'max_leaves': args.max_leaves,
        'max_depth': args.max_depth,
        'gamma': args.gamma,
        'colsample_bytree': args.colsample_bytree,
        'min_child_weight': args.min_child_weight,
    }

    model = XGBModel(
        lags = 12*24,
        lags_past_covariates = 33,
        output_chunk_length = 33,
        #to be tuned
        **tuned_params
    )

    print(f'Start fitting model')
    start = time.time()
    model.fit(ts_train_list_scaled, past_covariates=past_cov_train_list)
    print(f'End fitting after {time.time() - start} seconds')

    ##########################
    #<predict>
    ##########################

    start = pd.Timestamp('2014-04-01 14:00:00')

    all_forecasts = pd.DataFrame()
    IDs = df['ID'].unique()

    for i, id in enumerate(IDs):
        print(f'Forecasting for ID {id}, {i} of {len(IDs)}')

        forecast = pd.DataFrame()

        for step in range(11, 35):
            # start is 2pm
            print(step)
            fcst = model.historical_forecasts(
                series = ts_val_list_scaled[i],
                past_covariates=past_cov_val_list[i],
                stride=24, 
                start=start,
                forecast_horizon=step,
                retrain=False,
                verbose=True
            )
            # scale back
            fcst = scaler.inverse_transform(fcst)

            # convert to pandas
            fcst = fcst.pd_dataframe()
            # add to forecast
            forecast = pd.concat([forecast, fcst])

        forecast = forecast.reset_index()
        forecast.sort_values(by=['index'], inplace=True)
        forecast.rename(columns={'index': 'ds', 'y': 'yhat'}, inplace=True)
        forecast['ID'] = id
        all_forecasts = pd.concat([all_forecasts, forecast])

    print(all_forecasts.head())
    print(all_forecasts.tail())

    ##########################
    #<evaluate>
    ##########################
    # merge with df
    all_forecasts = pd.merge(all_forecasts, df, on=['ds', 'ID'], how='left')

    metrics = pd.DataFrame()

    for i, id in enumerate(IDs):
        forecast_id = all_forecasts[all_forecasts['ID'] == id]
        forecast_id['snaive'] = forecast_id['y'].shift(48)

        # calculate metrics
        rmse = _calc_rmse(predictions=forecast_id['yhat'], truth=forecast_id['y'])
        mae = _calc_mae(predictions=forecast_id['yhat'], truth=forecast_id['y'])
        nrmse = _calc_nrmse(predictions=forecast_id['yhat'], truth=forecast_id['y'])
        mape = _calc_mape(predictions=forecast_id['yhat'], truth=forecast_id['y'])
        mase = _calc_mase(predictions=forecast_id['yhat'], truth=forecast_id['y'], snaive_predictions=forecast_id['snaive'])
        msse = _calc_msse(predictions=forecast_id['yhat'], truth=forecast_id['y'], snaive_predictions=forecast_id['snaive'])

        # add to metrics
        metrics_row = pd.DataFrame({'ID': id, 'RMSE': rmse, 'MAE': mae, 'NRMSE': nrmse, 'MAPE': mape, 'MASE': mase, 'MSSE': msse}, index=[i])
        print(metrics_row)
        metrics = pd.concat([metrics, metrics_row])

    # log metrics
    run_id = Run.get_context(allow_offline=True).id

    with mlflow.start_run(run_id):
        # log parameters
        mlflow.log_params(tuned_params)

        #log metrics
        mlflow.log_metric("RMSE", metrics['RMSE'].mean())
        mlflow.log_metric("MAE", metrics['MAE'].mean())
        mlflow.log_metric("NRMSE", metrics['NRMSE'].mean())
        mlflow.log_metric("MAPE", metrics['MAPE'].mean())
        mlflow.log_metric("MASE", metrics['MASE'].mean())
        mlflow.log_metric("MSSE", metrics['MSSE'].mean())


    ##########################
    #<save forecasts and model>
    ##########################

    # save forecasts
    all_forecasts.to_csv(os.path.join(args.results, "forecasts_forecast_val.csv"), index=False)
    metrics.to_csv(os.path.join(args.results, "metrics_forecast_val.csv"), index=False)

if __name__ == "__main__":
    main()
