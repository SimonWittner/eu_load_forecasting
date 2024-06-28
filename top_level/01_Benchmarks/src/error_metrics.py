import numpy as np 
import pandas as pd

def _calc_mae(
    predictions: np.ndarray,
    truth: np.ndarray,
    truth_train: np.ndarray = None,
) -> float:
    """Calculates MAE error."""
    error_abs = np.abs(np.subtract(truth, predictions))
    return 1.0 * np.nanmean(error_abs, dtype="float32")

    
def _calc_mse(
    predictions: np.ndarray,
    truth: np.ndarray,
    truth_train: np.ndarray = None,
) -> float:
    """Calculates MSE error."""
    error_squared = np.square(np.subtract(truth, predictions))
    return 1.0 * np.nanmean(error_squared, dtype="float32")


def _calc_rmse(
    predictions: np.ndarray,
    truth: np.ndarray,
    truth_train: np.ndarray = None,
) -> float:
    """Calculates RMSE error."""
    mse = _calc_mse(predictions, truth)
    return np.sqrt(mse)

def _calc_mape(
    predictions: np.ndarray,
    truth: np.ndarray,
    truth_train: np.ndarray = None,
) -> float:
    """Calculates MAPE error."""
    error = np.subtract(truth, predictions)
    error_relative = np.abs(np.divide(error, truth))
    return 100.0 * np.nanmean(error_relative, dtype="float32")

def _calc_nrmse(
    predictions: np.ndarray,
    truth: np.ndarray,
    truth_train: np.ndarray = None,
) -> float:
    """Calculates NRMSE error."""
    mean_actuals = np.mean(truth)
    error = np.subtract(truth, predictions)
    normalized_error = np.divide(error,mean_actuals)
    return np.sqrt(np.nanmean(normalized_error**2,dtype="float32"))

# def _calc_mase(
#     predictions: np.ndarray,
#     truth: np.ndarray,
#     truth_train: np.ndarray,
# ) -> float:
#     """Calculates MASE error.
#     according to https://robjhyndman.com/papers/mase.pdf
#     Note: Naive error is computed over in-sample data.
#         MASE = MAE / NaiveMAE,
#     where: MAE = mean(|actual - forecast|)
#     where: NaiveMAE = mean(|actual_[i] - actual_[i-1]|) 
#     """
#     assert len(truth_train) > 1
#     mae = _calc_mae(predictions, truth)
#     naive_mae = _calc_mae(np.array(truth_train[:-1]), np.array(truth_train[1:]))
#     return np.divide(mae, 1e-9 + naive_mae)

def _seas_naive_fcst(forecasts,season,n_forecasts):
    
    forecasts_naive = pd.DataFrame(columns=[f"yhat{j}" for j in range(1,n_forecasts+1)])

    dates = forecasts["ds"].iloc[season : - n_forecasts + 1].reset_index(drop=True)
    # assemble last values based on season_length
    for i in range(len(dates)):
        last_season_vals = forecasts['y'].iloc[i : i + season].values
        yhat_row = [last_season_vals[j%season] for j in range(n_forecasts)]
        forecasts_naive.loc[len(forecasts_naive)] = yhat_row
    forecasts_naive['ds'] = dates
    forecasts_naive = pd.merge(left=forecasts.filter(['ds','y']), right=forecasts_naive, on='ds', how='left')

    # reshaping the naive forecasts to match the format of the neuralprophet forecasts (target-wise instead of origin-wise)
    cols = ["ds", "y"]  # cols to keep from df
    df_forecast = pd.concat((forecasts_naive[cols],), axis=1)
    yhat_values = forecasts_naive.loc[:, 'yhat1':f'yhat{n_forecasts}'].iloc[n_forecasts:-(n_forecasts-1)].to_numpy()
    yhat_array = np.reshape(yhat_values, (len(yhat_values), n_forecasts, 1))

    for forecast_lag in range(1, n_forecasts + 1):
        forecast = yhat_array[:, forecast_lag - 1, 0]
        pad_before = n_forecasts + forecast_lag - 1
        pad_after = n_forecasts - forecast_lag
        yhat = np.concatenate(([np.NaN] * pad_before, forecast, [np.NaN] * pad_after))
        name = f"yhat{forecast_lag}"
        df_forecast[name] = yhat
    forecasts_naive = df_forecast.copy()

    #To fill in the last rows. Each yhat row has the same value, so we are taking advantage of this. 
    forecasts_naive = forecasts_naive.fillna(method='bfill', axis=1)

    return forecasts_naive

def _calc_mase(
    predictions: np.ndarray,
    truth: np.ndarray,
    snaive_predictions: np.ndarray,
) -> float:
    """Calculates seasonal MASE error.
    according to https://robjhyndman.com/papers/mase.pdf
    Note: Naive error is computed over in-sample data.
        SMASE = MAE / SNaiveMAE,
    where: MAE = mean(|actual - forecast|)
    where: SNaiveMAE = mean(|actual_[i] - actual_[i-1]|) #seasonal naive als normierung nehmen 
    """
    mae = _calc_mae(predictions, truth)
    snaive_mae = _calc_mae(np.array(snaive_predictions), np.array(truth))
    return np.divide(mae, 1e-9 + snaive_mae)

def _calc_msse(
    predictions: np.ndarray,
    truth: np.ndarray,
    snaive_predictions: np.ndarray,
) -> float:
    """Calculates seasonal MASE error.
    according to https://robjhyndman.com/papers/mase.pdf
    Note: Naive error is computed over in-sample data.
        SMASE = MAE / SNaiveMAE,
    where: MAE = mean(|actual - forecast|)
    where: SNaiveMAE = mean(|actual_[i] - actual_[i-1]|) #seasonal naive als normierung nehmen 
    """
    mse = _calc_mse(predictions, truth)
    snaive_mse = _calc_mse(np.array(snaive_predictions), np.array(truth))
    return np.divide(mse, 1e-9 + snaive_mse)

def _calc_metrics(forecasts, start_forecast, n_forecasts, metric_names):
    """Calculates error metrics for the given forecasts.

    Args:
        forecasts: pd.DataFrame containing the forecasts
        start_forecast: int, first forecast to consider
        n_forecasts: int, number of forecasts to consider
        metric_names: list of str, metrics to compute

    Returns:
        pd.DataFrame containing the error metrics
        pd.DataFrame containing the error metrics averaged over all IDs
    """
    
    # Forecasts start at 2pm, so yhat1 corresponds to 3pm, yhat10 to midnight, yhat24 to 2pm
    hour_map = {1: 15, 2: 16, 3: 17, 4: 18, 5: 19, 6: 20, 7: 21, 8: 22,
            9: 23, 10: 0, 11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6,
            17: 7, 18: 8, 19: 9, 20: 10, 21: 11, 22: 12, 23: 13, 24: 14,
            25: 15, 26: 16, 27: 17, 28: 18, 29: 19, 30: 20, 31: 21, 32: 22, 33: 23}
    
    df_metrics_all_IDs = pd.DataFrame()
    
    # if ID is not in forecasts, then give dummy id to avoid error
    if 'ID' not in forecasts.columns:
        forecasts['ID'] = 'dummy'
        
    for ID, fcst_i in forecasts.groupby("ID"):
        # only compute naive forecast ig mase or msse is in metrics
        if 'MASE' in metric_names or 'MSSE' in metric_names:
            # naive forecasts (start slightly earlier than test period)
            forecasts_naive = _seas_naive_fcst(forecasts=fcst_i, season=24, n_forecasts=n_forecasts)

            # This is the timestamps where the test period starts, sorry for hard-coding
            forecasts_naive = forecasts_naive.iloc[-len(fcst_i):].reset_index(drop=True)

            # extract hour from ds column
            hour = forecasts['ds'].dt.hour

            for i in range(1, n_forecasts+1):
                # set non-matching values to NaN
                desired_hour = hour_map[i]
                fcst_i.loc[hour != desired_hour, f'yhat{i}'] = np.nan
                forecasts_naive.loc[hour != desired_hour, f'yhat{i}'] = np.nan
            
        df_metrics_one_ID = pd.DataFrame()
        for i in range(start_forecast,n_forecasts+1):
            preds = fcst_i[f"yhat{i}"].to_numpy(dtype="float64")
            if 'MASE' in metric_names or 'MSSE' in metric_names:
                preds_naive = forecasts_naive[f'yhat{i}'].to_numpy(dtype="float64")
            truth = fcst_i['y'].to_numpy(dtype="float64")

            metrics = pd.DataFrame()
            # only compute metrics that are mentioned in metric_names and add to metrics
            if 'RMSE' in metric_names:
                metrics = pd.concat((metrics,pd.DataFrame({"RMSE":_calc_rmse(predictions=preds, truth=truth)}, index=[i])))
            if 'MAE' in metric_names:
                metrics = pd.concat((metrics,pd.DataFrame({"MAE": _calc_mae(predictions=preds, truth=truth)}, index=[i])))
            if 'NRMSE' in metric_names:
                metrics = pd.concat((metrics,pd.DataFrame({"NRMSE": _calc_nrmse(predictions=preds, truth=truth)}, index=[i])))
            if 'MAPE' in metric_names:
                metrics = pd.concat((metrics,pd.DataFrame({"MAPE": _calc_mape(predictions=preds, truth=truth)}, index=[i])))
            if 'MASE' in metric_names:
                metrics = pd.concat((metrics,pd.DataFrame({"MASE": _calc_mase(predictions=preds, truth=truth, snaive_predictions=preds_naive)}, index=[i])))
            if 'MSSE' in metric_names:
                metrics = pd.concat((metrics,pd.DataFrame({"MSSE": _calc_msse(predictions=preds, truth=truth, snaive_predictions=preds_naive)}, index=[i])))

            df_metrics_one_ID = pd.concat((df_metrics_one_ID,metrics)) # contains all yhats 
            df_metrics_one_ID['ID'] = ID
        
        df_metrics_all_IDs = pd.concat((df_metrics_all_IDs,df_metrics_one_ID))

    # Mean across all the prediction steps (yhats) per ID
    df_metrics_all_IDs = df_metrics_all_IDs.groupby(['ID']).mean().reset_index()
    print("saving new metrics successful")
    return df_metrics_all_IDs, df_metrics_all_IDs.mean().to_frame().T #, mae_per_step