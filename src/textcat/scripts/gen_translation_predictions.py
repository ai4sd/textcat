"""
Run inference pipeline with provided model on validation dataset.

Predictions are stored as single Pandas dataframe 'val_predictions.parquet'.
All validation data are included (n=99854), anomalies as well.
splits are distinguished in the df by the 'split' column.
"""

import argparse
import os

import pandas as pd
from transformers import BartForConditionalGeneration
import torch
from torch.utils.data import DataLoader

from textcat.ml.dataset.translation_dataset import AdsorptionTranslationDataset
from textcat.ml.tokenizer.adsorption_tokenizer import AdsorptionTokenizer
from textcat.ml.tokenizer.bert_tokenizer import GeneralTokenizer
from textcat.ml.train_utils import run_inference_translation, get_last_checkpoint


def main():
    """
    Run inference pipeline on OC20 validation data for BART translation (IS2RS) models.
    """
    parser = argparse.ArgumentParser(description="Run inference pipeline for OC20 data.")
    parser.add_argument('--model_dir', type=str, help="Directory containing model checkpoint and vocabulary.")
    parser.add_argument("--x", type=str, help="Name of the columns containing the sequence.")
    parser.add_argument("--y", type=str, help="Name of the columns containing the target sequence.")
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
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--infer_train_dataset", action='store_true', default=False, help='If set, run forward pass on training dataset.')
    parser.add_argument("--infer_val_datasets", action='store_true', default=False, help='If set, run forward pass on validation datasets.')

    args = parser.parse_args()    

    X, Y = args.x, args.y
    DF_COLUMNS = [X, Y, 'anomaly']
    MODEL_PATH = args.model_dir
    BATCH_SIZE = args.batch_size

    tokenizer = GeneralTokenizer(
        vocab_file=MODEL_PATH+"/vocabulary.txt",
        basic_tokenizer=AdsorptionTokenizer(args.remove_tilde, args.xval),
        remove_mapping=True,
    )

    checkpoint = get_last_checkpoint(MODEL_PATH)
    print("Loading model at ", checkpoint)
    model = BartForConditionalGeneration.from_pretrained(
            checkpoint, local_files_only=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    model.to(device)
    c1 = lambda x: os.path.exists(x)
    c2 = lambda x: len(pd.read_parquet(x)) == len(df_val_id) + len(df_val_ood_ads) + len(df_val_ood_cat) + len(df_val_ood_both)

    if args.infer_val_datasets:
        OUTPUT_VAL_DF_NAME = MODEL_PATH + "/val_predictions.parquet"
        df_val_id = pd.read_parquet(args.val_data_path + "val_id/data.parquet", engine="fastparquet", columns=DF_COLUMNS)  
        df_val_ood_cat = pd.read_parquet(args.val_data_path + "val_ood_cat/data.parquet", engine="fastparquet", columns=DF_COLUMNS)  
        df_val_ood_ads = pd.read_parquet(args.val_data_path + "val_ood_ads/data.parquet", engine="fastparquet", columns=DF_COLUMNS)  
        df_val_ood_both = pd.read_parquet(args.val_data_path + "val_ood_both/data.parquet", engine="fastparquet", columns=DF_COLUMNS)  
        
        if c1(OUTPUT_VAL_DF_NAME) and c2(OUTPUT_VAL_DF_NAME):
            print("File exists. Exiting the program.")
            return None
        
        if args.remove_val_anomalies:
            for anomaly_id in args.remove_val_anomalies:     
                df_val_id = df_val_id[df_val_id['anomaly'] != anomaly_id]
                df_val_ood_ads = df_val_ood_ads[df_val_ood_ads['anomaly'] != anomaly_id]
                df_val_ood_cat = df_val_ood_cat[df_val_ood_cat['anomaly'] != anomaly_id]
                df_val_ood_both = df_val_ood_both[df_val_ood_both['anomaly'] != anomaly_id]
                print(f"Discarded anomaly {anomaly_id} data from validation sets")

        val_set_id = AdsorptionTranslationDataset(df_val_id[X], df_val_id[Y], tokenizer, args.max_len, xval_encoding=args.xval)
        val_set_ood_ads = AdsorptionTranslationDataset(df_val_ood_ads[X], df_val_ood_ads[Y], tokenizer, args.max_len, xval_encoding=args.xval)  
        val_set_ood_cat = AdsorptionTranslationDataset(df_val_ood_cat[X], df_val_ood_cat[Y], tokenizer, args.max_len, xval_encoding=args.xval)  
        val_set_ood_both = AdsorptionTranslationDataset(df_val_ood_both[X], df_val_ood_both[Y], tokenizer, args.max_len, xval_encoding=args.xval)  

        loader_id = DataLoader(val_set_id, batch_size=BATCH_SIZE, shuffle=False)
        loader_ood_ads = DataLoader(val_set_ood_ads, batch_size=BATCH_SIZE, shuffle=False)
        loader_ood_cat = DataLoader(val_set_ood_cat, batch_size=BATCH_SIZE, shuffle=False)
        loader_ood_both = DataLoader(val_set_ood_both, batch_size=BATCH_SIZE, shuffle=False)

        print(f"Start inference for validation sets on {device}")
        val_id_pred, val_ood_ads_pred, val_ood_cat_pred, val_ood_both_pred = [], [], [], []
        run_inference_translation(loader_id, device, model, tokenizer, val_id_pred)
        run_inference_translation(loader_ood_ads, device, model, tokenizer, val_ood_ads_pred)
        run_inference_translation(loader_ood_cat, device, model, tokenizer, val_ood_cat_pred)
        run_inference_translation(loader_ood_both, device, model, tokenizer, val_ood_both_pred)

        df_val_id['pred'] = val_id_pred
        df_val_ood_ads['pred'] = val_ood_ads_pred
        df_val_ood_cat['pred'] = val_ood_cat_pred
        df_val_ood_both['pred'] = val_ood_both_pred
        df_val_id['split'] = 'id'
        df_val_ood_ads['split'] = 'ads'
        df_val_ood_cat['split'] = 'cat'
        df_val_ood_both['split'] = 'both'
        df = pd.concat([df_val_id, df_val_ood_ads, df_val_ood_cat, df_val_ood_both])
        df.to_parquet(OUTPUT_VAL_DF_NAME)

    if args.infer_train_dataset:
        df_train = pd.read_parquet(args.train_data_path, engine="fastparquet", columns=DF_COLUMNS)  
        if args.remove_train_anomalies:
            for anomaly_id in args.remove_train_anomalies:     
                df_train = df_train[df_train['anomaly'] != anomaly_id]
                print(f"Discarded anomaly {anomaly_id} data from training set")
        train_set = AdsorptionTranslationDataset(df_train[X], df_train[Y], tokenizer, args.max_len, xval_encoding=args.xval)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)
        train_predictions = []
        print(f"Start inference for training set on {device}")
        run_inference_translation(train_loader, device, model, tokenizer, train_predictions)
        df_train['pred'] = train_predictions
        df_train.to_parquet(MODEL_PATH + "/train_predictions.parquet")

if __name__ == '__main__':
    main()