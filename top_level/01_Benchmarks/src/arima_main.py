import argparse
import pandas as pd
import numpy as np
import os
import time

from darts import TimeSeries
from darts.models import ARIMA
from darts.utils.missing_values import fill_missing_values


def main():
    """Main function of the script."""

    # input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--id", type=str, help="id of the time series")
        
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

    # shift Temp_fcst and ds_fcst by 33 hours to match actual time of the forecast for future covariates

    #df['Temp_fcst'] = df['Temp_fcst'].shift(33)
    #df['ds_fcst'] = df['ds_fcst'].shift(33)

    df['ds'] = pd.to_datetime(df['ds'])
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.rename(columns={'country': 'ID'})
    df['temp'] = pd.to_numeric(df['temp'], errors='coerce')

    df = df[['ds', 'ID', 'temp', 'y']]

    ################## 
    #<train the model>
    ##################
    id = args.id

    df_ID = df[df['ID'] == id]
    print(df_ID.head())
    print(df_ID.tail())
 
    ts = TimeSeries.from_dataframe(df_ID, time_col='ds', value_cols=['y'])
    ts = fill_missing_values(ts)
    ts_train, ts_test = ts.split_before(pd.Timestamp('2014-01-01'))

    future_cov = TimeSeries.from_dataframe(df_ID, time_col='ds', value_cols=['temp'])
    future_cov = fill_missing_values(future_cov)
    future_cov_train, future_cov_test = future_cov.split_before(pd.Timestamp('2014-01-01'))

    print(f'Start fitting model for ID {id}')
    model = ARIMA(p=1, d=0, q=1, seasonal_order=(1, 0, 1, 24), trend='ct')
    #model = ARIMA(p=1, d=2, q=1, seasonal_order=(1, 0, 1, 24), trend='ct')
    model.fit(ts_train, future_covariates=future_cov_train)
    print(f'Finished fitting model for ID {id}')

    start = pd.Timestamp('2014-01-01 14:00:00')
    forecast = pd.DataFrame()

    for step in range(11, 35):
        # start is 2pm
        fcst = model.historical_forecasts(
            series = ts, 
            future_covariates=future_cov,
            stride=24, 
            start=start,
            forecast_horizon=step,
            retrain=False,
            verbose=False
        )
        # convert to pandas
        fcst = fcst.pd_dataframe()

        # add to forecast
        forecast = pd.concat([forecast, fcst])

    forecast.sort_values(by=['time'], inplace=True)
    forecast.reset_index(inplace=True)
    forecast.rename(columns={'time': 'ds', 'y': 'yhat'}, inplace=True)

    forecast['ID'] = id

    print(forecast.head())
    print(forecast.tail())

    ##########################
    #<save forecasts and model>
    ##########################

    # save forecasts
    forecast.to_csv(os.path.join(args.results, "forecasts_forecast_test.csv"), index=False)

if __name__ == "__main__":
    main()
