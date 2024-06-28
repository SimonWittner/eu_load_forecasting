import argparse
import pandas as pd
import numpy as np
import os
import time

from darts.dataprocessing.transformers import Scaler
from darts.utils.missing_values import fill_missing_values
from darts.timeseries import TimeSeries
from darts.models import XGBModel


def main():
    """Main function of the script."""

    # input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
        
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
    past_cov_list = []
    past_cov_train_list = []
    IDs = df['ID'].unique()

    for id in IDs:
        df_ID = df[df['ID'] == id]
        ts = TimeSeries.from_dataframe(df_ID, time_col='ds', value_cols=['y'])
        ts = fill_missing_values(ts)
        ts_train, ts_test = ts.split_before(pd.Timestamp('2014-01-01'))

        past_cov = TimeSeries.from_dataframe(df_ID, time_col='ds', value_cols=['temp'])
        past_cov_train, past_cov_test = past_cov.split_before(pd.Timestamp('2014-01-01'))

        ts_list.append(ts)
        ts_train_list.append(ts_train)
        past_cov_list.append(past_cov)
        past_cov_train_list.append(past_cov_train)

    print('Starting Scaling')
    scaler = Scaler()
    ts_list_scaled = scaler.fit_transform(ts_list)
    ts_train_list_scaled = scaler.fit_transform(ts_train_list)

    # best parameters
    best_params = {
        "learning_rate": 0.123,
        "subsample": 0.861,
        "max_leaves": 20,
        "max_depth": 30,
        "gamma": 0.00039,
        "colsample_bytree": 0.874,
        "min_child_weight": 7,
    }

    model = XGBModel(
        lags = 15*24,
        lags_past_covariates = 33,
        output_chunk_length = 33,
        **best_params
    )

    print(f'Start fitting model')
    start = time.time()
    model.fit(ts_train_list_scaled, past_covariates=past_cov_train_list)
    print(f'End fitting after {time.time() - start} seconds')

    start = pd.Timestamp('2014-01-01 14:00:00')
    all_forecasts = pd.DataFrame()
    IDs = df['ID'].unique()

    for i, id in enumerate(IDs):
        print(f'Forecasting for ID {id}, {i} of {len(IDs)}')

        forecast = pd.DataFrame()

        for step in range(11, 35):
            # start is 2pm
            print(step)
            fcst = model.historical_forecasts(
                series = ts_list_scaled[i],
                past_covariates=past_cov_list[i],
                stride=24, 
                start=start,
                forecast_horizon=step,
                retrain=False,
                verbose=True
            )
            # convert to pandas
            fcst = fcst.pd_dataframe()
            # add to forecast
            forecast = pd.concat([forecast, fcst])
        forecast = forecast.reset_index()
        forecast.sort_values(by=['index'], inplace=True)
        forecast.rename(columns={'index': 'ds', 'y': 'yhat'}, inplace=True)
        forecast['ID'] = id
        all_forecasts = pd.concat([all_forecasts, forecast])

    
    # scale back: convert to ts, put into list and scale back
    all_forecasts_ts = []
    for id in IDs:
        df_ID = all_forecasts[all_forecasts['ID'] == id]
        ts = TimeSeries.from_dataframe(df_ID, time_col='ds', value_cols=['yhat'])
        all_forecasts_ts.append(ts)

    all_forecasts_ts_scaled_back = scaler.inverse_transform(all_forecasts_ts)

    # convert back to pandas
    all_forecasts = pd.DataFrame()
    for i, id in enumerate(IDs):
        df_ID = all_forecasts_ts_scaled_back[i].pd_dataframe()
        df_ID['ID'] = id
        all_forecasts = pd.concat([all_forecasts, df_ID])
    
    all_forecasts.reset_index(inplace=True)
    
    print(all_forecasts.head())
    print(all_forecasts.tail())

    ##########################
    #<save forecasts and model>
    ##########################

    # save forecasts
    all_forecasts.to_csv(os.path.join(args.results, "forecasts_forecast_test.csv"), index=False)

if __name__ == "__main__":
    main()
