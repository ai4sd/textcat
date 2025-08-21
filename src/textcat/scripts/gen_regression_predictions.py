"""
Run Inference pipeline given a model checkpoint and OC20 datasets.

Predictions are stored as Pandas dataframes in .parquet format.
"""

import argparse

import pandas as pd
from transformers import BertForSequenceClassification
import torch
from torch.utils.data import DataLoader

from textcat.ml.dataset.regression_dataset import AdsorptionRegressionDataset
from textcat.ml.tokenizer.adsorption_tokenizer import AdsorptionTokenizer
from textcat.ml.tokenizer.bert_tokenizer import GeneralTokenizer
from textcat.ml.numbert_wrapper import NumBertForSequenceClassification
from textcat.ml.train_utils import run_inference_regression, get_last_checkpoint


def main():
    """
    Run inference pipeline on OC20 validation sets for BERT regression models.
    """
    parser = argparse.ArgumentParser(description="Run inference pipeline for OC20 data.")
    parser.add_argument('--model_dir', type=str, help="Directory containing model checkpoint, vocab, and scaler.")
    parser.add_argument("--x", type=str, help="Name of the columns containing the sequence.")
    parser.add_argument("--y", type=str, help="Name of the column containing the labels.")
    parser.add_argument("--train_data_path", type=str, default="data/dataframes/train/data.parquet")
    parser.add_argument("--val_data_path", type=str, default="data/dataframes/")
    parser.add_argument("--xval", action='store_true', default=False, help='Enable number encoding for bond lengths in SMILES. Not available for now.')
    parser.add_argument("--remove_tilde", action='store_true', default=False, help='Remove "~" from SMILES before tokenization.')
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--remove_train_anomalies", type=int, nargs='+', default=None,
                        choices=[1, 2, 3, 4], 
                        help="Filter out data with specified anomalies from training set.")
    parser.add_argument("--remove_val_anomalies", type=int, nargs='+', default=None,
                        choices=[1, 2, 3, 4], 
                        help="Filter out data with specified anomalies from validation sets.")

    args = parser.parse_args()

    X, Y = args.x, args.y
    DF_COLUMNS = [X, Y, 'anomaly']

    # Load train and validation dataframes (requires fastparquet installed!)
    df_train = pd.read_parquet(args.train_data_path, engine="fastparquet", columns=DF_COLUMNS)    
    df_val_id = pd.read_parquet(args.val_data_path + "val_id/data.parquet", engine="fastparquet", columns=DF_COLUMNS)  
    df_val_ood_cat = pd.read_parquet(args.val_data_path + "val_ood_cat/data.parquet", engine="fastparquet", columns=DF_COLUMNS)  
    df_val_ood_ads = pd.read_parquet(args.val_data_path + "val_ood_ads/data.parquet", engine="fastparquet", columns=DF_COLUMNS)  
    df_val_ood_both = pd.read_parquet(args.val_data_path + "val_ood_both/data.parquet", engine="fastparquet", columns=DF_COLUMNS)  

    # Remove data with known anomalies
    if args.remove_train_anomalies:
        for anomaly_id in args.remove_train_anomalies:
            df_train = df_train[df_train['anomaly'] != anomaly_id]
            print(f"Neglect anomaly {anomaly_id} data from training set")
    if args.remove_val_anomalies:
        for anomaly_id in args.remove_val_anomalies:     
            df_val_id = df_val_id[df_val_id['anomaly'] != anomaly_id]
            df_val_ood_ads = df_val_ood_ads[df_val_ood_ads['anomaly'] != anomaly_id]
            df_val_ood_cat = df_val_ood_cat[df_val_ood_cat['anomaly'] != anomaly_id]
            df_val_ood_both = df_val_ood_both[df_val_ood_both['anomaly'] != anomaly_id]
            print(f"Neglect anomaly {anomaly_id} data from validation sets")

    # Load scaler, vocabulary, and tokenizer configs
    scaler = torch.load(args.model_dir+"/scaler.pt")

    # Load tokenizer
    tokenizer = GeneralTokenizer(
        vocab_file=args.model_dir+"/vocabulary.txt",
        basic_tokenizer=AdsorptionTokenizer(args.remove_tilde, args.xval),
        remove_mapping=True,
    )

    checkpoint = get_last_checkpoint(args.model_dir)
    print(f"Using {checkpoint} for inference")

    if args.xval:
        model = NumBertForSequenceClassification.from_pretrained(
            args.model_dir+checkpoint, num_labels=1, local_files_only=True)
    else:
        model = BertForSequenceClassification.from_pretrained(
            args.model_dir_checkpoint, num_labels=1, local_files_only=True)
    train_set = AdsorptionRegressionDataset(df_train[X], df_train[Y], tokenizer, args.max_len, True, args.xval)      
    val_set_id = AdsorptionRegressionDataset(df_val_id[X], df_val_id[Y], tokenizer, args.max_len, train_set.scaler, args.xval)
    val_set_ood_ads = AdsorptionRegressionDataset(df_val_ood_ads[X], df_val_ood_ads[Y], tokenizer, args.max_len, train_set.scaler, args.xval)
    val_set_ood_cat = AdsorptionRegressionDataset(df_val_ood_cat[X], df_val_ood_cat[Y], tokenizer, args.max_len, train_set.scaler, args.xval)
    val_set_ood_both = AdsorptionRegressionDataset(df_val_ood_both[X], df_val_ood_both[Y], tokenizer, args.max_len, train_set.scaler, args.xval)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    model.to(device)
    
    BATCH_SIZE = 256
    loader_train = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)  # Keep shuffle=False to mantain order
    loader_id = DataLoader(val_set_id, batch_size=BATCH_SIZE, shuffle=False)
    loader_ood_ads = DataLoader(val_set_ood_ads, batch_size=BATCH_SIZE, shuffle=False)
    loader_ood_cat = DataLoader(val_set_ood_cat, batch_size=BATCH_SIZE, shuffle=False)
    loader_ood_both = DataLoader(val_set_ood_both, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Start inference on {device}")
    train_pred, val_id_pred, val_ood_ads_pred, val_ood_cat_pred, val_ood_both_pred = [], [], [], [], []
    run_inference_regression(loader_train, device, model, train_pred, scaler)
    run_inference_regression(loader_id, device, model, val_id_pred, scaler)
    run_inference_regression(loader_ood_ads, device, model, val_ood_cat_pred, scaler)
    run_inference_regression(loader_ood_cat, device, model, val_ood_cat_pred, scaler)
    run_inference_regression(loader_ood_both, device, model, val_ood_both_pred, scaler)

    df_train['pred'] = train_pred
    df_train.to_parquet(args.model_dir + "/train_predictions.parquet")

    df_val_id['pred'] = val_id_pred
    df_val_ood_ads['pred'] = val_ood_ads_pred
    df_val_ood_cat['pred'] = val_ood_cat_pred
    df_val_ood_both['pred'] = val_ood_both_pred
    df_val_id['split'] = 'id'
    df_val_ood_ads['split'] = 'ads'
    df_val_ood_cat['split'] = 'cat'
    df_val_ood_both['split'] = 'both'
    df_val = pd.concat([df_val_id, df_val_ood_ads, df_val_ood_cat, df_val_ood_both])
    df_val.to_parquet(args.model_dir + "/val_predictions.parquet")

if __name__ == '__main__':
    main()