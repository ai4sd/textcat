"""
Script for generating fingerprints from a finetuned BERT model for regression.
Requires a forward pass for all data (train and val). Fingerprints will be stored as 
Numpy arrays in the same folder of the finetuned model.

Note: Fingerprint here refers to the pooler output from BERT encoder.
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
from transformers import BertForSequenceClassification
from textcat.ml.numbert_wrapper import NumBertForSequenceClassification
import torch
from torch.utils.data import DataLoader

from textcat.ml.dataset.regression_dataset import AdsorptionRegressionDataset
from textcat.ml.tokenizer.adsorption_tokenizer import AdsorptionTokenizer
from textcat.ml.tokenizer.bert_tokenizer import GeneralTokenizer
from textcat.ml.train_utils import get_fingerprints


def main():
    """
    Finetune BERT model for regression of adsorption energy ("eads", scalar)
    from SMILES with OC20 dataset.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help="Finetuned model directory. Must be in 'models/finetuning/' and have the format model_name/checkpoint-XXXXX.")
    parser.add_argument("--x", type=str, help="Name of the columns containing the sequence.")
    parser.add_argument("--y", type=str, default="eads_eV", choices=["eads_eV", "scaled_eads_eV"], help="Name of the column containing the labels.")
    parser.add_argument("--train_data_path", type=str, default="data/dataframes/train/data.parquet", help='Dataframe in .parquet format containing the training data.')
    parser.add_argument("--val_data_path", type=str, default="data/dataframes/")
    parser.add_argument("--max_len", type=int, default=192)
    parser.add_argument("--remove_tilde", action='store_true', default=False, help='Remove "~" from SMILES before tokenization.')
    parser.add_argument("--xval", action='store_true', default=False, help='Enable number encoding for bond lengths in SMILES.')
    parser.add_argument("--remove_train_anomalies", type=int, nargs='+', default=None,
                        choices=[1, 2, 3, 4], 
                        help="Filter out data with specified anomalies from training set.")
    parser.add_argument("--remove_val_anomalies", type=int, nargs='+', default=None,
                        choices=[1, 2, 3, 4], 
                        help="Filter out data with specified anomalies from validation sets.")
    args = parser.parse_args()
    print(args)

    # Define paths for model, files, etc. 
    PRETRAINED_MODEL_PATH = "models/finetuning/" + args.model_path
    VOCAB = PRETRAINED_MODEL_PATH + "/vocabulary.txt"
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
            print(f"Removed anomaly {anomaly_id} data from training set")
    if args.remove_val_anomalies:
        for anomaly_id in args.remove_val_anomalies:     
            df_val_id = df_val_id[df_val_id['anomaly'] != anomaly_id]
            df_val_ood_ads = df_val_ood_ads[df_val_ood_ads['anomaly'] != anomaly_id]
            df_val_ood_cat = df_val_ood_cat[df_val_ood_cat['anomaly'] != anomaly_id]
            df_val_ood_both = df_val_ood_both[df_val_ood_both['anomaly'] != anomaly_id]
            print(f"Removed anomaly {anomaly_id} data from validation sets")

    # Load tokenizer
    tokenizer = GeneralTokenizer(
        vocab_file=VOCAB,
        basic_tokenizer=AdsorptionTokenizer(args.remove_tilde, args.xval),
        remove_mapping=True,
    )
    
    checkpoint_folders = [f for f in os.listdir(PRETRAINED_MODEL_PATH) if os.path.isdir(os.path.join(PRETRAINED_MODEL_PATH, f))]
    step_pattern = re.compile(r"checkpoint-(\d+)")
    checkpoint = "/" + max(checkpoint_folders, key=lambda folder: int(step_pattern.search(folder).group(1)) if step_pattern.search(folder) else -1)
    print(f"Using {checkpoint} for inference")
    # Instantiate model, train and val datasets
    if args.xval:  # xval is for numerical encoding
        model = NumBertForSequenceClassification.from_pretrained(
            PRETRAINED_MODEL_PATH + checkpoint, num_labels=1)
    else:
        model = BertForSequenceClassification.from_pretrained(
            PRETRAINED_MODEL_PATH + checkpoint, num_labels=1)
    train_set = AdsorptionRegressionDataset(df_train[X], df_train[Y], tokenizer, args.max_len, True, args.xval)      
    val_set_id = AdsorptionRegressionDataset(df_val_id[X], df_val_id[Y], tokenizer, args.max_len, train_set.scaler, args.xval)
    val_set_ood_ads = AdsorptionRegressionDataset(df_val_ood_ads[X], df_val_ood_ads[Y], tokenizer, args.max_len, train_set.scaler, args.xval)
    val_set_ood_cat = AdsorptionRegressionDataset(df_val_ood_cat[X], df_val_ood_cat[Y], tokenizer, args.max_len, train_set.scaler, args.xval)
    val_set_ood_both = AdsorptionRegressionDataset(df_val_ood_both[X], df_val_ood_both[Y], tokenizer, args.max_len, train_set.scaler, args.xval)

    # INFERENCE PIPELINE
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    BATCH_SIZE = 256
    loader_train = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)  # shuffle=False to keep order
    loader_id = DataLoader(val_set_id, batch_size=BATCH_SIZE, shuffle=False)
    loader_ood_ads = DataLoader(val_set_ood_ads, batch_size=BATCH_SIZE, shuffle=False)
    loader_ood_cat = DataLoader(val_set_ood_cat, batch_size=BATCH_SIZE, shuffle=False)
    loader_ood_both = DataLoader(val_set_ood_both, batch_size=BATCH_SIZE, shuffle=False)

    with torch.no_grad():
        train_fps = get_fingerprints(loader_train, device, model)
        val_id_fps = get_fingerprints(loader_id, device, model)
        val_ood_ads_fps = get_fingerprints(loader_ood_ads, device, model)
        val_ood_cat_fps = get_fingerprints(loader_ood_cat, device, model)
        val_ood_both_fps = get_fingerprints(loader_ood_both, device, model)

    np.savez(PRETRAINED_MODEL_PATH + "/train_bert_fingerprints.npz", fp=np.array(train_fps))
    np.savez(PRETRAINED_MODEL_PATH + "/val_id_bert_fingerprints.npz", fp=np.array(val_id_fps))
    np.savez(PRETRAINED_MODEL_PATH + "/val_ood_ads_bert_fingerprints.npz", fp=np.array(val_ood_ads_fps))
    np.savez(PRETRAINED_MODEL_PATH + "/val_ood_cat_bert_fingerprints.npz", fp=np.array(val_ood_cat_fps))
    np.savez(PRETRAINED_MODEL_PATH + "/val_ood_both_bert_fingerprints.npz", fp=np.array(val_ood_both_fps))

if __name__ == "__main__": 
    main()