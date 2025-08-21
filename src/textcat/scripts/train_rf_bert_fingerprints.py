"""
Script for training multiple model types in scikit-learn
(random forest, gradient boosting, XGBoost) using the BERT fingerprints
from finetuned BERT regression models.
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


def main():
    """
    Train sklearn regression models on BERT pooler output fingerprints.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp_path', type=str, help="Model directory. Must be in 'models/finetuning/' and contain train and val data in .npz format.")
    parser.add_argument("--x", type=str, help="Name of the columns containing the sequence.")
    parser.add_argument("--y", type=str, default="eads_eV", choices=["eads_eV", "scaled_eads_eV"], help="Name of the column containing the labels.")
    parser.add_argument("--train_data_path", type=str, default="data/dataframes/train/data.parquet", help='Dataframe in .parquet format containing the training data.')
    parser.add_argument("--val_data_path", type=str, default="data/dataframes/")
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument("--remove_train_anomalies", type=int, nargs='+', default=None,
                        choices=[1, 2, 3, 4], 
                        help="Filter out data with specified anomalies from training set.")
    parser.add_argument("--remove_val_anomalies", type=int, nargs='+', default=None,
                        choices=[1, 2, 3, 4], 
                        help="Filter out data with specified anomalies from validation sets.")
    parser.add_argument('--model_type', type=str, default='rf', choices=['rf', 'gb', 'xgb'])

    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--n_jobs', type=int, default=1)
    args = parser.parse_args()
    print(args)


    # Load train and validation dataframes (requires fastparquet installed!)
    X, Y = args.x, args.y
    DF_COLUMNS = [X, Y, 'anomaly', 'ads_energy_eV']
    df_train = pd.read_parquet(args.train_data_path, engine="fastparquet", columns=DF_COLUMNS)    
    df_val_id = pd.read_parquet(args.val_data_path + "val_id/data.parquet", engine="fastparquet", columns=DF_COLUMNS)  
    df_val_ood_cat = pd.read_parquet(args.val_data_path + "val_ood_cat/data.parquet", engine="fastparquet", columns=DF_COLUMNS)  
    df_val_ood_ads = pd.read_parquet(args.val_data_path + "val_ood_ads/data.parquet", engine="fastparquet", columns=DF_COLUMNS)  
    df_val_ood_both = pd.read_parquet(args.val_data_path + "val_ood_both/data.parquet", engine="fastparquet", columns=DF_COLUMNS)  

    # Remove data with known anomalies
    if args.remove_train_anomalies:
        for anomaly_id in args.remove_train_anomalies:
            df_train = df_train[df_train['anomaly'] != anomaly_id]
            print(f"Removed anomaly {anomaly_id} data from training set")
    if args.remove_val_anomalies:
        for anomaly_id in args.remove_val_anomalies:     
            df_val_id = df_val_id[df_val_id['anomaly'] != anomaly_id]
            df_val_ood_ads = df_val_ood_ads[df_val_ood_ads['anomaly'] != anomaly_id]
            df_val_ood_cat = df_val_ood_cat[df_val_ood_cat['anomaly'] != anomaly_id]
            df_val_ood_both = df_val_ood_both[df_val_ood_both['anomaly'] != anomaly_id]
            print(f"Removed anomaly {anomaly_id} data from validation sets")

    # Load fingerprints and labels
    train_x = np.load(args.fp_path + '/train_bert_fingerprints.npz', allow_pickle=True)['fp']
    val_id_x = np.load(args.fp_path + '/val_id_bert_fingerprints.npz', allow_pickle=True)['fp']
    val_ood_ads_x = np.load(args.fp_path + '/val_ood_ads_bert_fingerprints.npz', allow_pickle=True)['fp']
    val_ood_cat_x = np.load(args.fp_path + '/val_ood_cat_bert_fingerprints.npz', allow_pickle=True)['fp']
    val_ood_both_x = np.load(args.fp_path + '/val_ood_both_bert_fingerprints.npz', allow_pickle=True)['fp']

    train_y = df_train[Y].values.tolist()
    val_id_y = df_val_id[Y].values.tolist()
    val_ood_ads_y = df_val_ood_ads[Y].values.tolist()
    val_ood_cat_y = df_val_ood_cat[Y].values.tolist()
    val_ood_both_y = df_val_ood_both[Y].values.tolist()

    print("Data loaded!")

    if args.model_type == 'rf':    
        model = RandomForestRegressor(args.n_estimators, 
                                    criterion='squared_error',
                                    n_jobs = args.n_jobs, 
                                    verbose=2) 
                                    # max_features=256)
    elif args.model_type == 'gb':
        model = GradientBoostingRegressor(loss='squared_error', 
                                          n_estimators=args.n_estimators,
                                          learning_rate=0.1, 
                                          verbose=2,  
                                          )
    elif args.model_type == 'xgb':
        model = XGBRegressor(n_jobs=args.n_jobs, 
                             device=args.device, 
                             verbose=2, 
                             learning_rate=0.1, 
                             objective='reg:squarederror', 
                             n_estimators=args.n_estimators, 
                             max_depth=10,
                             gamma=0.05)
    else:
        raise Exception()

    print("Model loaded!")

    model.fit(train_x, train_y)

    print("RF model trained!")
    val_id_yy = model.predict(val_id_x)
    val_ood_ads_yy = model.predict(val_ood_ads_x)
    val_ood_cat_yy = model.predict(val_ood_cat_x)
    val_ood_both_yy = model.predict(val_ood_both_x)
    
    # Get MAE for validation sets
    mae_id = mean_absolute_error(val_id_y, val_id_yy)
    mae_ood_ads = mean_absolute_error(val_ood_ads_y, val_ood_ads_yy)
    mae_ood_cat = mean_absolute_error(val_ood_cat_y, val_ood_cat_yy)
    mae_ood_both = mean_absolute_error(val_ood_both_y, val_ood_both_yy)
    mae_tot = (mae_id + mae_ood_ads + mae_ood_cat + mae_ood_both) / 4

    print(mae_id)
    print(mae_ood_ads)
    print(mae_ood_cat)
    print(mae_ood_both)
    print(mae_tot)
    # model.save_model(args.output_path)

if __name__ == "__main__": 
    main()