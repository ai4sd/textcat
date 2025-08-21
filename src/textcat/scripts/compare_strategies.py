"""
This scripts aims at comparing the performance of two different strategies
to predict the adsorption energy from sequence text representations of 
adsorption structures (i.e., molecules interacting with surfaces).

Strategy 1: Direct regression SMILES (initial guess state) -> Energy (Target). 1 model 
Strategy 2: Translation and regression SMILES (initial guess state) -> SMILES (Optimized state) -> Energy.

Both strategies are compared starting from the same representation. Comparison is made
in terms of regression accuracy (MAE, RMSE, R2, MDAE).

Notes:
- Translation with xVal encoding not implemented.
"""

import argparse
import time

import pandas as pd
from rdkit import RDLogger
from transformers import BertForSequenceClassification, BartForConditionalGeneration

from textcat.ml.nn import AdsorptionPredictor
from textcat.ml.train_utils import get_last_checkpoint
RDLogger.DisableLog('rdApp.*')

def main():
    """
    Compare strategies for adsorption energy prediction with OC20 data.
    """
    parser = argparse.ArgumentParser(description="Compare energy prediction strategies.")
    parser.add_argument("--x", type=str, help="Name of the column containing the text (SMILES) representing the initial state.")
    parser.add_argument('--is2re_model', type=str, help="Directory containing regression model checkpoint, vocab, scaler for initial state to relaxed energy (strategy 1).")
    parser.add_argument('--is2rs_model', type=str, help="Directory containing translation model checkpoint, vocab, scaler for initial state to relaxed state (strategy 2).")
    parser.add_argument('--rs2re_model', type=str, help="Directory containing regression model checkpoint, vocab, scaler for initial state to relaxed energy (strategy 1).")
    parser.add_argument('--output', type=str, help="Name of generated folder with results (df with predictions)")
    parser.add_argument("--val_data_path", type=str, default="data/dataframes/")
    parser.add_argument("--xval", action='store_true', default=False, help='Enable number encoding for bond lengths in SMILES. Not available for now.')
    parser.add_argument("--remove_tilde", action='store_true', default=False, help='Remove "~" from SMILES before tokenization.')
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--remove_val_anomalies", type=int, nargs='+', default=None,
                        choices=[1, 2, 3, 4], 
                        help="Filter out data with specified anomalies from validation sets.")
    args = parser.parse_args()

    X, Y = args.x, "eads_eV"
    DF_COLUMNS = [X, Y, 'anomaly']

    # Load train and validation dataframes (requires fastparquet installed!)
    df_val_id = pd.read_parquet(args.val_data_path + "val_id/data.parquet", engine="fastparquet", columns=DF_COLUMNS)  
    df_val_ood_cat = pd.read_parquet(args.val_data_path + "val_ood_cat/data.parquet", engine="fastparquet", columns=DF_COLUMNS)  
    df_val_ood_ads = pd.read_parquet(args.val_data_path + "val_ood_ads/data.parquet", engine="fastparquet", columns=DF_COLUMNS)  
    df_val_ood_both = pd.read_parquet(args.val_data_path + "val_ood_both/data.parquet", engine="fastparquet", columns=DF_COLUMNS) 
    df_val_id['type'] = 'id'
    df_val_ood_ads['type'] = 'ads'
    df_val_ood_cat['type'] = 'cat'
    df_val_ood_both['type'] = 'both'
    df = pd.concat([df_val_id, df_val_ood_ads, df_val_ood_cat, df_val_ood_both], axis=0, ignore_index=True)

    print("DATA LOADED!")
    # Remove data with known anomalies
    if args.remove_val_anomalies:
        for anomaly_id in args.remove_val_anomalies:     
            df = df[df['anomaly'] != anomaly_id]
            print(f"Neglect anomaly {anomaly_id} data from validation sets")

    # IS2RE MODEL (Initial Structure to Relaxed Energy)
    is2re = BertForSequenceClassification.from_pretrained(
            get_last_checkpoint(args.is2re_model), num_labels=1, local_files_only=True)
    # IS2RS MODEL (Initial Structure to Relaxed Structure)
    is2rs = BartForConditionalGeneration.from_pretrained(get_last_checkpoint(args.is2rs_model), local_files_only=True)
    # RS2RE MODEL (Relaxed Structure to Relaxed Energy)
    rs2re = BertForSequenceClassification.from_pretrained(
        get_last_checkpoint(args.rs2re_model), num_labels=1, local_files_only=True) 
     
    print("MODELS LOADED!")

    s1_model = AdsorptionPredictor(seq2num_model=is2re, 
                                   vocab_path=args.is2re_model+"/vocabulary.txt", 
                                   remove_tilde=args.remove_tilde, 
                                   xval=args.xval, 
                                   scaler=args.is2re_model+"/scaler.pt")
    s2_model = AdsorptionPredictor(seq2seq_model=is2rs, 
                                   seq2num_model=rs2re, 
                                   vocab_path=args.is2rs_model+"/vocabulary.txt", 
                                   remove_tilde=args.remove_tilde, 
                                   xval=args.xval, 
                                   scaler=args.rs2re_model+"/scaler.pt")

    print("Start Strategy 1 evaluation!!! (Direct Regression)")
    t0 = time.time()
    y1 = s1_model(df[X].to_list())
    t1 = time.time() - t0
    t0 = time.time()
    print("Start Strategy 2 evaluation!!! (Translation + Regression)")
    y2 = s2_model(df[X].to_list()) 
    t2 = time.time() - t0  

    df['y1'] = y1['eads_eV']
    df['y2'] = y2['eads_eV']
    df['err1'] = df[Y] - df['y1']
    df['err2'] = df[Y] - df['y2']
    df['abs_err1'] = df['err1'].abs()
    df['abs_err2'] = df['err2'].abs()
    df['smiles2'] = y2['rs']
    df.to_parquet(args.output)

    print(df.groupby('type')['abs_err1'].mean())
    print(df.groupby('type')['abs_err2'].mean())

    print("Validation time (Strategy 1): ", t1)
    print("Validation time (Strategy 2): ", t2)

if __name__ == '__main__':
    main()