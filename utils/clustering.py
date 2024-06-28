import numpy as np 
import pandas as pd
from tsfeatures import tsfeatures, stl_features, entropy, stability, lumpiness
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import datetime
import matplotlib.pyplot as plt

# ML AZURE
from azure.ai.ml import MLClient
from azure.ai.ml import command, Input, Output
from azure.identity import DefaultAzureCredential
from azureml.core import Workspace, Dataset, Datastore

def mapping_tsfeatures(df, normalise=True, freq=24, calulate_tsfeatures=True, calculate_em=True):
    """
    Calculate various time series measures for dataframe in long format with columns ds, y, and unique_id.

    Args:
    df: Pandas Dataframe
        Data frame in long format with columns ds, y, and unique_id.
    normalise: Boolean 
        indicating whether to normalise the data.
    freq: Integer 
        indicating the frequency of the time series. FREQS = {'H': 24, 'D': 1, 'M': 12, 'Q': 4, 'W':1, 'Y': 1}
    calulate_tsfeatures: Boolean
        indicating whether to calculate tsfeatures or not.
    calculate_em: Boolean
        indicating whether to calculate energy metrics or not.

    Returns: Pandas Dataframe 
        with time series measures as columns and unique_id as index.
    """
    # rename column ID to unique_id
    df = df.rename(columns={'ID': 'unique_id'})
    df['ds'] = pd.to_datetime(df['ds'])

    # fill NaN: forwardfill and backwardfill
    df = df.groupby('unique_id', group_keys=False).apply(lambda x: x.ffill())
    df = df.groupby('unique_id', group_keys=False).apply(lambda x: x.bfill())

    if normalise:
        # Normalise between 0 and 1
        df['y'] = df.groupby('unique_id')['y'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
        #df['y'] = df.groupby('unique_id')['y'].apply(lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0)

    ### Calculate ts-features
    if calulate_tsfeatures:
        print('Starting tsfeatures computation ...')
        ts_features = tsfeatures(df, freq = 24, features = [stl_features, entropy, stability, lumpiness])
        print('Finished tsfeatures computation')

    ### Calculate custom measures
    measures = pd.DataFrame()
    measures['var'] = df.groupby('unique_id')['y'].var()

    if calculate_em:
        print('Starting energy-metrics computation ...')
        ## Calculate features June
        # Eday - mean of daily energy use
        temp = df.groupby(['unique_id', df['ds'].dt.date])['y'].mean().reset_index()
        eday = temp.groupby('unique_id')['y'].mean().reset_index()
        eday = eday.rename(columns={'y': 'eday'})
        measures = pd.merge(measures, eday, on='unique_id', how='left')

        # Ehour - mean of hourly energy use
        temp = df.groupby(['unique_id', df['ds'].dt.hour])['y'].mean().reset_index()
        ehour = temp.groupby('unique_id')['y'].mean().reset_index()
        ehour = ehour.rename(columns={'y': 'ehour'})
        measures = pd.merge(measures, ehour, on='unique_id', how='left')

        # Epeak - mean energy use of peak hour in a day
        highest_y_per_date = df.groupby(['unique_id', df['ds'].dt.date])['y'].max().reset_index()
        epeak = highest_y_per_date.groupby('unique_id')['y'].mean().reset_index()
        epeak = epeak.rename(columns={'y': 'epeak'})
        measures = pd.merge(measures, epeak, on='unique_id', how='left')     

        # Ebase - mean base energy use of a day - TODO: validate
        highest_y_per_date = df.groupby(['unique_id', df['ds'].dt.date])['y'].max().reset_index()
        lowest_y_per_date = df.groupby(['unique_id', df['ds'].dt.date])['y'].min().reset_index()
        lowest_y_per_date['ebase'] = (highest_y_per_date['y'] - lowest_y_per_date['y']) / 2
        ebase = lowest_y_per_date.groupby('unique_id')['ebase'].mean().reset_index()
        measures = pd.merge(measures, ebase, on='unique_id', how='left')

        # Emin - mean of minimum energy use of a day    
        lowest_y_per_date = df.groupby(['unique_id', df['ds'].dt.date])['y'].min().reset_index()
        emin = lowest_y_per_date.groupby('unique_id')['y'].mean().reset_index()
        emin = emin.rename(columns={'y': 'emin'})
        measures = pd.merge(measures, emin, on='unique_id', how='left')

        # Enight
        filtered_df = df[(df['ds'].dt.hour >= 2) & (df['ds'].dt.hour <= 6)]
        enight = filtered_df.groupby('unique_id')['y'].sum().reset_index()
        enight = enight.rename(columns={'y': 'enight'})
        measures = pd.merge(measures, enight, on='unique_id', how='left')
    
        # Emorning
        filtered_df = df[(df['ds'].dt.hour >= 6) & (df['ds'].dt.hour <= 10)]
        emorning = filtered_df.groupby('unique_id')['y'].sum().reset_index()
        emorning = emorning.rename(columns={'y': 'emorning'})
        measures = pd.merge(measures, emorning, on='unique_id', how='left')

        # Enoon
        filtered_df = df[(df['ds'].dt.hour >= 10) & (df['ds'].dt.hour <= 14)]
        enoon = filtered_df.groupby('unique_id')['y'].sum().reset_index()
        enoon = enoon.rename(columns={'y': 'enoon'})
        measures = pd.merge(measures, enoon, on='unique_id', how='left')

        # Eafternoon
        filtered_df = df[(df['ds'].dt.hour >= 14) & (df['ds'].dt.hour <= 18)]
        eafternoon = filtered_df.groupby('unique_id')['y'].sum().reset_index()
        eafternoon = eafternoon.rename(columns={'y': 'eafternoon'})
        measures = pd.merge(measures, eafternoon, on='unique_id', how='left')       

        # Eevening
        filtered_df = df[(df['ds'].dt.hour >= 18) & (df['ds'].dt.hour <= 22)]
        eevening = filtered_df.groupby('unique_id')['y'].sum().reset_index()
        eevening = eevening.rename(columns={'y': 'eevening'})
        measures = pd.merge(measures, eevening, on='unique_id', how='left')

        # Emidnight
        filtered_df = df[(df['ds'].dt.hour >= 22) | (df['ds'].dt.hour <= 2)]
        emidnight = filtered_df.groupby('unique_id')['y'].sum().reset_index()
        emidnight = emidnight.rename(columns={'y': 'emidnight'})
        measures = pd.merge(measures, emidnight, on='unique_id', how='left')

        print('Finished energy-metrics computation')

    # add features of tsfeatures
    if calulate_tsfeatures:
        measures = pd.merge(measures, ts_features, on='unique_id', how='left')

    # rename column unique_id to ID
    ID_tsmeasures = measures.rename(columns={'unique_id': 'ID'})

    print('Finished energy-metrics computation')
    return ID_tsmeasures

def mapping_energy_metrics(df, normalise=True):
    """
    Calculate various energy metrics measures for dataframe in long format with columns ds, y, and unique_id.

    Args:
    df: Pandas Dataframe
        Data frame in long format with columns ds, y, and unique_id.
    normalise: Boolean 
        indicating whether to normalise the data.
    freq: Integer 
        indicating the frequency of the time series. FREQS = {'H': 24, 'D': 1, 'M': 12, 'Q': 4, 'W':1, 'Y': 1}

    Returns: Pandas Dataframe 
        with energy_metrics as columns and unique_id as index.
    """
    print('Start')
    # rename column ID to unique_id
    df = df.rename(columns={'ID': 'unique_id'})
    df['ds'] = pd.to_datetime(df['ds'])

    # fill NaN: forwardfill and backwardfill
    df = df.groupby('unique_id', group_keys=False).apply(lambda x: x.ffill())
    df = df.groupby('unique_id', group_keys=False).apply(lambda x: x.bfill())

    if normalise:
        print('Starting normalisation ...')
        # Normalise between 0 and 1
        df['y'] = df.groupby('unique_id')['y'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
        #df['y'] = df.groupby('unique_id')['y'].apply(lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0)    

    ### Calculate custom measures
    print('Starting energy-metrics computation ...')
    measures = pd.DataFrame()
    measures['var'] = df.groupby('unique_id')['y'].var()

    ## Calculate features June
    # Eday - mean of daily energy use
    temp = df.groupby(['unique_id', df['ds'].dt.date])['y'].mean().reset_index()
    eday = temp.groupby('unique_id')['y'].mean().reset_index()
    eday = eday.rename(columns={'y': 'eday'})
    measures = pd.merge(measures, eday, on='unique_id', how='left')

    # Ehour - mean of hourly energy use
    temp = df.groupby(['unique_id', df['ds'].dt.hour])['y'].mean().reset_index()
    ehour = temp.groupby('unique_id')['y'].mean().reset_index()
    ehour = ehour.rename(columns={'y': 'ehour'})
    measures = pd.merge(measures, ehour, on='unique_id', how='left')

    # Epeak - mean energy use of peak hour in a day
    highest_y_per_date = df.groupby(['unique_id', df['ds'].dt.date])['y'].max().reset_index()
    epeak = highest_y_per_date.groupby('unique_id')['y'].mean().reset_index()
    epeak = epeak.rename(columns={'y': 'epeak'})
    measures = pd.merge(measures, epeak, on='unique_id', how='left')     

    # Ebase - mean base energy use of a day - TODO: validate
    highest_y_per_date = df.groupby(['unique_id', df['ds'].dt.date])['y'].max().reset_index()
    lowest_y_per_date = df.groupby(['unique_id', df['ds'].dt.date])['y'].min().reset_index()
    lowest_y_per_date['ebase'] = (highest_y_per_date['y'] - lowest_y_per_date['y']) / 2
    ebase = lowest_y_per_date.groupby('unique_id')['ebase'].mean().reset_index()
    measures = pd.merge(measures, ebase, on='unique_id', how='left')

    # Emin - mean of minimum energy use of a day    
    lowest_y_per_date = df.groupby(['unique_id', df['ds'].dt.date])['y'].min().reset_index()
    emin = lowest_y_per_date.groupby('unique_id')['y'].mean().reset_index()
    emin = emin.rename(columns={'y': 'emin'})
    measures = pd.merge(measures, emin, on='unique_id', how='left')

    # Enight
    filtered_df = df[(df['ds'].dt.hour >= 6) & (df['ds'].dt.hour <= 10)]
    enight = filtered_df.groupby('unique_id')['y'].sum().reset_index()
    enight = enight.rename(columns={'y': 'enight'})
    measures = pd.merge(measures, enight, on='unique_id', how='left')
 
    # Emorning
    filtered_df = df[(df['ds'].dt.hour >= 6) & (df['ds'].dt.hour <= 10)]
    emorning = filtered_df.groupby('unique_id')['y'].sum().reset_index()
    emorning = emorning.rename(columns={'y': 'emorning'})
    measures = pd.merge(measures, emorning, on='unique_id', how='left')

    # Enoon
    filtered_df = df[(df['ds'].dt.hour >= 10) & (df['ds'].dt.hour <= 14)]
    enoon = filtered_df.groupby('unique_id')['y'].sum().reset_index()
    enoon = enoon.rename(columns={'y': 'enoon'})
    measures = pd.merge(measures, enoon, on='unique_id', how='left')

    # Eafternoon
    filtered_df = df[(df['ds'].dt.hour >= 14) & (df['ds'].dt.hour <= 18)]
    eafternoon = filtered_df.groupby('unique_id')['y'].sum().reset_index()
    eafternoon = eafternoon.rename(columns={'y': 'eafternoon'})
    measures = pd.merge(measures, eafternoon, on='unique_id', how='left')       

    # Eevening
    filtered_df = df[(df['ds'].dt.hour >= 18) & (df['ds'].dt.hour <= 22)]
    eevening = filtered_df.groupby('unique_id')['y'].sum().reset_index()
    eevening = eevening.rename(columns={'y': 'eevening'})
    measures = pd.merge(measures, eevening, on='unique_id', how='left')

    # Emidnight
    filtered_df = df[(df['ds'].dt.hour >= 22) | (df['ds'].dt.hour <= 2)]
    emidnight = filtered_df.groupby('unique_id')['y'].sum().reset_index()
    emidnight = emidnight.rename(columns={'y': 'emidnight'})
    measures = pd.merge(measures, emidnight, on='unique_id', how='left')

    # Ewholeday
    ewholeday = df.groupby('unique_id')['y'].sum().reset_index()
    ewholeday = ewholeday.rename(columns={'y': 'ewholeday'})
    measures = pd.merge(measures, ewholeday, on='unique_id', how='left')

    print('Finished absolute energy-metrics computation')

    ## Relative Metrics
    # Rbase
    measures['rbase'] = measures['ebase'] / measures['eday']

    # Rminmax
    measures['rminmax'] = measures['emin'] / measures['epeak']

    # Rm2w
    measures['rm2w'] = measures['emorning'] / measures['ewholeday']

    # Rn2w
    measures['rn2w'] = measures['enoon'] / measures['ewholeday']

    # Re2w
    measures['re2w'] = measures['eevening'] / measures['ewholeday']

    # Rni2w
    measures['rni2w'] = measures['enight'] / measures['ewholeday']

    ID_energy_metrics = measures.rename(columns={'unique_id': 'ID'})

    return ID_energy_metrics

def clustering(ID_feature, df, n_cluster=3, normalise=True, n_pca=1, elbow=1, upload=False, df_test=None):
    """
    Calculates clusters of IDs.

    Args:
    ID_feature: Pandas Dataframe
        Feature vector. IDs as index and features as columns.
    df: Pandas Dataframe
        data frame in long format with columns ID, ds, y.
    n_cluster: Integer for number of clusters.
    n_pca: Integer for number of principal components.
    elbow: Integer for number of clusters to plot in elbow plot.
    normalise: Boolean 
        indicating whether to normalise the data.
    upload: Boolean 
        indicating whether to upload the data.
    df_test: Pandas Dataframe
        data frame with test data, only requiqred if upload=True. 


    Returns: Pandas Dataframe 
        with clusters as columns and unique_id as index.

    """
    ID_copy = pd.DataFrame(ID_feature['ID'])
    features = ID_feature.drop(columns=['ID'])

    # normalise
    if normalise:
        for col in features:
            if (features[col].max() != features[col].min()).any():
                ## Normizing
                #features[col] = (features[col] - features[col].min()) / (features[col].max() - features[col].min())
                ## Standardizing
                features[col] = (features[col] - features[col].mean()) / features[col].std()
            else:
                features = features.drop(columns=[col])
                print(f'Feature {col} was dropped because it is constant.')
                continue
    print('Finished normalisation/standardisation')
    
    # pca
    if n_pca > 1:
        print(f'Starting PCA with {n_pca} components ...')
        pca = PCA(n_components=n_pca)
        pca_df = pca.fit_transform(features)
        pca_df = pd.DataFrame(pca_df)
        for col in pca_df.columns:
            pca_df.rename(columns={col: f'PC{col+1}'}, inplace=True)
        features = pca_df
    else:
        print('Starting clustering without PCA')

    # Elbow Plot - n_clusters
    if elbow > 1:
        inertia = []
        e = elbow
        for i in range(1,e):
            kmeans = KMeans(n_clusters=i, random_state=0).fit(features)
            inertia.append(kmeans.inertia_)
        plt.figure(figsize=(4,4))
        plt.plot(range(1,e), inertia, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Sum of Squared Distances (Inertia)')
        plt.show()

    # clustering
    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(features)
    features['cluster'] = kmeans.labels_

    # merge with ID
    features_cluster = pd.merge(ID_copy, features, left_index=True, right_index=True)

    for i in features_cluster['cluster'].value_counts().index:
        count = features_cluster['cluster'].value_counts()[i]
        print(f'Cluster {i} has {count} IDs')

    if upload:
        ### Upload to datastore
        from azure.ai.ml.entities import Data
        from azure.ai.ml.constants import AssetTypes
        import time
        # Get a handle to the workspace
        from azureml.core import Workspace, Dataset, Datastore
        import mlflow
        subscription_id = '89c1a2ce-18f3-4f74-ae22-ffbfc9da1a6f'
        resource_group = 'rg-east-ml'
        workspace_name = 'machinelearning-ws'
        workspace = Workspace(subscription_id, resource_group, workspace_name)
        datastore = Datastore.get(workspace, "workspaceblobstore")

        # authenticate
        credential = DefaultAzureCredential()

        # Get a handle to the workspace
        ml_client = MLClient(
            credential=credential,
            subscription_id = subscription_id,
            resource_group_name = resource_group,
            workspace_name = workspace_name
        )

        workspace = Workspace(subscription_id, resource_group, workspace_name)
        datastore = Datastore.get(workspace, "workspaceblobstore")
        today = datetime.date.today().strftime("%Y-%m-%d")

        for i in features_cluster['cluster'].unique():
            count = features_cluster['cluster'].value_counts()[i]
            name = f'cluster_{i}_{today}_{count}s'
            ID_cluster = features_cluster[features_cluster['cluster']==i]['ID'].values
            df_cluster = df[df['ID'].isin(ID_cluster)]
            df_test_cluster = df_test[df_test['ID'].isin(ID_cluster)]
            # concat
            df_cluster = pd.concat([df_cluster, df_test_cluster], ignore_index=True)
            # save as csv
            df_cluster.to_csv(f'predata/{name}.csv', index=False)

            path = f'predata/{name}.csv'
            description = f'Clustered train-test-data for {name}'
            v1 = time.strftime("%Y.%m.%d.%H%M%S", time.gmtime())

            my_data = Data(
            name = name,
            version = v1,
            description = description,
            path = path,
            type = AssetTypes.URI_FILE,
            )

            ml_client.data.create_or_update(my_data)
            print(f"Data asset created. Name: {my_data.name}, version: {my_data.version}")

    return features_cluster

def get_results(name):
    """
    Gets results from forecasting on Azure.

    Args:
    name: String
        Name of job on Azure as string.

    Returns: Pandas Dataframe 
        Results of forecasting.

    """
    import torch
    #from neuralprophet import NeuralProphet
    # ML AZURE
    # Get a handle to the workspace
    from azureml.core import Workspace, Dataset, Datastore
    import mlflow
    subscription_id = '89c1a2ce-18f3-4f74-ae22-ffbfc9da1a6f'
    resource_group = 'rg-east-ml'
    workspace_name = 'machinelearning-ws'
    workspace = Workspace(subscription_id, resource_group, workspace_name)
    datastore = Datastore.get(workspace, "workspaceblobstore")

    path = f'azureml/{name}/results/'

    # load np model
    #model = Dataset.File.from_files(path=(datastore, f'azureml/{name}/results/model.np'))
    #model.download(target_path='.', overwrite=True)
    #m = torch.load("model.np", map_location=torch.device('cpu'))

    # test forecast
    file = 'forecasts_forecast_test.csv'
    dataset = Dataset.Tabular.from_delimited_files(path=(datastore, path + file))
    forecast_test = dataset.to_pandas_dataframe() 
    forecast_test['ds'] = pd.to_datetime(forecast_test['ds'])

    # test metrics
    file = 'metrics_metrics_test.csv'
    dataset = Dataset.Tabular.from_delimited_files(path=(datastore, path + file))
    metrics_test = dataset.to_pandas_dataframe()

    # yhat
    yhat_ID_test = add_day_ahead_column(forecast_test, 'yhat')

    return forecast_test, metrics_test, yhat_ID_test

def add_day_ahead_column(df, name):
    """Add a new column to the DataFrame with the values of the column to be fetched
    from a given date at 2pm, the forecast is for the next day, 24 hours, starting at midnight

    Args:
        df (pd.DataFrame): DataFrame with the forecast
        name (str): name of the column to extract (eg 'yhat'); for quantiles, use '1.5%' etc

    Returns:
        df (pd.DataFrame): DataFrame with the new column
    """

    def choose_column(row):
        hour = row['ds'].hour
        if '%' not in name:
            row_name = f"{name}{hour + 10}"
        else: #consider naming of quantiles, eg yhat1 1.5%
            row_name = f"yhat{hour + 10} {name}"
        return row[row_name]
    
    df[f"{name}_day_ahead"] = df.apply(choose_column, axis=1)
    return df

def sum_until_threshold(df, df_ID, threshold_value, threshold_global, threshold_cluster, threshold_country, agg_no):
    """Function to add up the IDs (-nodes) until a threshold is reached.

    Args:
        df (pd.DataFrame): DataFrame with ds, y, ID (df)
        df_ID (pd.DataFrame): DataFrame with ID, y_mean of ID (df_magnitude)
        threshold_value (float): Either global value or threshold_quantile to sum until (0.25, 0.75 etc.)
        threshold_global: True or False, whether to use a global threshold for all country & cluster or not
        threshold_cluster: True or False, whether to use a threshold for each cluster or not
        threshold_country: True or False, whether to use a threshold for each country or not
        agg_no (int): number of aggregations 

    Returns:
        df (pd.DataFrame): DataFrame with summed IDs in it
    """
    mapping = pd.DataFrame()

    # Check if 'country' is a name of a column
    if 'country' in df_ID.columns:
        print('country is a column in df_ID')
    else:
        df_ID = df_ID.rename(columns={'country': 'country'})
        df = df.rename(columns={'country': 'country'})

    for agg_no in range(agg_no):
        print("\n\nstarting aggregation no.", agg_no)
        ID_to_add = pd.DataFrame()
        mapping_temp = pd.DataFrame()

        for i in df_ID["cluster"].unique():
            df_ID_cluster = df_ID[df_ID["cluster"] == i]

            for j in df_ID_cluster["country"].unique():
                df_ID_cluster_country = df_ID_cluster[df_ID_cluster["country"] == j]

                # Sort the DataFrame by 'y_mean' column in ascending order
                #sorted_df = df_ID.sort_values(by='y_mean')
                sorted_df = df_ID_cluster_country.sort_values(by='y_mean')

                # Initialize variables
                cumulative_sum = 0
                selected_rows = []

                ### Calculate the threshold -- TODO: check how aggregated IDs influence quantile in 3.
                # 1. Global threshold (for all countrys and clusters)
                if threshold_global:
                    threshold = threshold_value
                # 2. Threshold for each cluster
                elif threshold_cluster:
                    condition = threshold_value['cluster'] == i
                    threshold = threshold_value.loc[condition, 'threshold'].values[0]
                # 3. Threshold for each country
                elif threshold_country:
                    condition = threshold_value['country'] == j
                    threshold = threshold_value.loc[condition, 'threshold'].values[0]
                # 4. Threshold quantile for each country & cluster
                else:
                    threshold = df_ID_cluster_country['y_mean'].quantile(threshold_value)

                # Iterate through sorted DataFrame
                for index, row in sorted_df.iterrows():
                    cumulative_sum += row['y_mean']
                    selected_rows.append(row)
                    # Check if cumulative sum exceeds the threshold
                    if cumulative_sum >= threshold:
                        break
                    
                # Create a new DataFrame from the selected rows
                tempo_dataframe = pd.DataFrame(selected_rows)
                #ID_to_add = ID_to_add.append(tempo_dataframe).reset_index(drop=True)
                ID_to_add = pd.concat([ID_to_add, tempo_dataframe], ignore_index=True)

                # delete all rows with ID in temp_dataframe from df_ID
                df_ID = df_ID[~df_ID["ID"].isin(tempo_dataframe["ID"])]

                # Add aggregated ID to df_ID
                #tempo_dataframe["ID"] = str(j) + "_cluster0" + str(i) + "_agg"
                #df_ID = df_ID.append(tempo_dataframe).reset_index(drop=True)

        # remove duplicates
        ID_to_add = ID_to_add.drop_duplicates(subset=['ID', 'cluster'])

        # Count IDs and interrupt aggregation if only one ID is left
        total_unique_ids_count = ID_to_add['ID'].nunique()

        if total_unique_ids_count > 1:
            # Info
            #print(f'-nodes per Cluster that will be aggregated in this round {agg_no}:\n', ID_to_add["cluster"].value_counts())
            print('\033[1m' + f'-nodes per Cluster that will be aggregated in this round {agg_no}:\n' + '\033[0m', '\033[1m' + f'{ID_to_add.groupby("cluster").size().sort_index().to_frame().T}\n' + '\033[0m')
            print(f'Total count of each combination of country and Cluster:\n', ID_to_add.groupby(["country", "cluster"]).size().unstack(fill_value=0))
            #print('For each cluster 10 aggregated Nodes (for each country) are resulting instead of the number above!')

            ### Aggregation
            print("Starting Aggregation of Buses...")

            #v2
            # find all IDs of ID_to_add in df
            filtered_df = df[df["ID"].isin(ID_to_add["ID"])]

            # groupby country, cluster, ds and sum y
            grouped_df = filtered_df.groupby(['country', 'cluster', 'ds'])['y'].sum().reset_index()

            # rename ID of grouped df
            grouped_df['ID'] = grouped_df['country'].astype(str) + '_cluster0' + grouped_df['cluster'].astype(str) + '_agg_no_' + str(agg_no)

            # Mapping for which IDs get aggregated
            mapping_temp['ID_list'] = filtered_df.groupby(['country', 'cluster'])['ID'].transform(lambda x: ';'.join(x.unique().astype(str)))
            mapping_temp['cluster'] = filtered_df.groupby(['country', 'cluster'])['cluster'].transform(lambda x: ';'.join(x.unique().astype(str)))
            mapping_temp['country'] = filtered_df.groupby(['country', 'cluster'])['country'].transform(lambda x: ';'.join(x.unique().astype(str)))
            mapping_temp['no_of_agg_IDs'] = filtered_df.groupby(['country', 'cluster'])['ID'].transform('nunique')
            mapping_temp['aggregation_no'] = agg_no
            mapping_temp = mapping_temp.drop_duplicates(subset=['ID_list'])

            # delete all rows with ID in ID_to_add from df
            df = df[~df["ID"].isin(ID_to_add["ID"])]    

            # add grouped_df to df
            #df = df.append(grouped_df).reset_index(drop=True)
            df = pd.concat([df, grouped_df], ignore_index=True)

            print("Finished Aggregation of Buses")

            #mapping = mapping.append(mapping_temp).reset_index(drop=True)
            mapping = pd.concat([mapping, mapping_temp], ignore_index=True)

        else:
            print('\n\n\nOnly one ID left, no further aggregation possible')
            break

    return ID_to_add, df, mapping