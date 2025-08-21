"""
Script for finetuning BERT model for regression task with OC20 data.
Include inference pipeline with best model for train and validation sets at the end. 

Note:
- args.val_sample allows to monitor the metrics on subsets of the original validation sets
  during training. However, the final inference pipeline is run on the whole validation sets.
"""

import argparse
import os
import shutil

import pandas as pd
from torch import nn, save
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from textcat.ml.numbert_wrapper import NumBertForSequenceClassification
import torch
from torch.utils.data import DataLoader

from textcat.ml.dataset.regression_dataset import AdsorptionRegressionDataset
from textcat.ml.tokenizer.adsorption_tokenizer import AdsorptionTokenizer
from textcat.ml.tokenizer.bert_tokenizer import GeneralTokenizer
from textcat.ml.train_utils import run_inference_regression, compute_metrics_regression_w, MAE, get_last_checkpoint


def main():
    """
    Finetune BERT model for regression of adsorption energy
    from SMILES with OC20 dataset.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--model_path', type=str, help="Pretrained model directory. Must be in 'models/regression/'.")
    parser.add_argument("--x", type=str, help="Name of the columns containing the sequence.")
    parser.add_argument("--y", type=str, default="eads_eV", choices=["eads_eV", "scaled_eads_eV"], help="Name of the column containing the labels.")
    parser.add_argument("--vocab_path", type=str, help="Vocabulary .txt file. Must be in 'data/vocabularies'.", default='vocabulary.txt')
    parser.add_argument("--output_path", type=str, default="models/regression/", help='Folder where the model will be stored.')
    parser.add_argument("--train_data_path", type=str, default="data/dataframes/train/data.parquet", help='Dataframe in .parquet format containing the training data.')
    parser.add_argument("--val_data_path", type=str, default="data/dataframes/")
    parser.add_argument("--num_epochs", type=float, default=25)
    parser.add_argument("--learning_rate", type=float, default=0.0000025)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=176)
    parser.add_argument("--eval_steps", type=int, default=2500)
    parser.add_argument("--save_steps", type=int, default=2500)
    parser.add_argument("--logging_steps", type=int, default=2500)
    parser.add_argument("--remove_tilde", action='store_true', default=False, help='Remove "~" from SMILES before tokenization.')
    parser.add_argument("--xval", action='store_true', default=False, help='Enable number encoding for bond lengths in SMILES.')
    parser.add_argument("--remove_train_anomalies", type=int, nargs='+', default=None,
                        choices=[1, 2, 3, 4], 
                        help="Filter out data with specified anomalies from training set.")
    parser.add_argument("--remove_val_anomalies", type=int, nargs='+', default=None,
                        choices=[1, 2, 3, 4], 
                        help="Filter out data with specified anomalies from validation sets.")
    parser.add_argument("--mae_loss", action='store_true', default=False,
                        help="Set MAE as loss function instead of MSE (default)") 
    parser.add_argument("--resume_from_checkpoint", action='store_true', default=False)
    parser.add_argument("--augment_online", action='store_true', default=False, help='Train with enumerated SMILES instead of canonical SMILES only.')  
    parser.add_argument("--augment_offline", default=None, help='Provide column name to double dataset size by adding cases when IS==RS. Only valid when input is initial state.')
    parser.add_argument("--train_val_with_corrupted", default=None, type=str, help="Use corrupted SMILES from translation model as train and validation input data. Provide path to .parquet dataframes with train and val predictions.")
    parser.add_argument("--metric_for_best", type=str, default="ood_ads_MAE_eV")
    parser.add_argument("--val_sample", type=float, default=None, help="Perform validation on a fraction (0-1) ubset of val_sample entries instead of the entire dataset.")
    args = parser.parse_args()
    print(args)

    # Define paths for model, files, and tokenizer
    VOCAB = os.path.join("data/vocabularies", args.vocab_path)
    OUTPUT_DIR = args.output_path + args.run_name
    PRETRAINED_MODEL_PATH = get_last_checkpoint(os.path.join("models/regression/", args.model_path))
    X, Y = args.x, args.y
    DF_COLUMNS = [X, Y, 'anomaly']
    if args.augment_offline:
        DF_COLUMNS.append(args.augment_offline)

    tokenizer = GeneralTokenizer(
        vocab_file=VOCAB,
        basic_tokenizer=AdsorptionTokenizer(args.remove_tilde, args.xval),
        remove_mapping=True,
    )

    # Load train and validation dataframes (requires fastparquet installed!)
    df_train = pd.read_parquet(args.train_data_path, engine="fastparquet", columns=DF_COLUMNS)    
    df_val_id = pd.read_parquet(args.val_data_path + "val_id/data.parquet", engine="fastparquet", columns=DF_COLUMNS)  
    df_val_ood_cat = pd.read_parquet(args.val_data_path + "val_ood_cat/data.parquet", engine="fastparquet", columns=DF_COLUMNS)  
    df_val_ood_ads = pd.read_parquet(args.val_data_path + "val_ood_ads/data.parquet", engine="fastparquet", columns=DF_COLUMNS)  
    df_val_ood_both = pd.read_parquet(args.val_data_path + "val_ood_both/data.parquet", engine="fastparquet", columns=DF_COLUMNS)  

    # Remove data with known anomalies and/or reduce validation sets
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

    if args.train_val_with_corrupted:
        print(f"Training and evaluating on SMILES predicted by translation model {args.train_val_with_corrupted}!")
        train_path = args.train_val_with_corrupted + '/train_pred.parquet'
        corrupted_smiles_train = pd.read_parquet(train_path, engine="fastparquet")['pred']
        df_train[X] = corrupted_smiles_train  #[:int(len(corrupted_smiles_train)/2)]
        if len(df_train[X]) != len(corrupted_smiles_train):
            print("CAREFUL!", len(corrupted_smiles_train), len(df_train))
        if 'val_predictions.parquet' in os.listdir(args.train_val_with_corrupted):
            val_path = args.train_val_with_corrupted + '/val_predictions.parquet'
            df_val = pd.read_parquet(val_path, engine="fastparquet")
            df_val_id[X] = df_val[df_val['split'] == 'id']['pred']
            df_val_ood_ads[X] = df_val[df_val['split'] == 'ads']['pred']
            df_val_ood_cat[X] = df_val[df_val['split'] == 'cat']['pred']
            df_val_ood_both[X] = df_val[df_val['split'] == 'both']['pred']
        else:
            try:
                val_path = os.path.join(args.train_val_with_corrupted, 'val_')
                df_val_id[X] = pd.read_parquet(val_path + "id_pred.parquet", engine="fastparquet")['pred']  
                df_val_ood_cat[X] = pd.read_parquet(val_path + "ood_cat.parquet", engine="fastparquet")['pred']    
                df_val_ood_ads[X] = pd.read_parquet(val_path + "ood_ads_pred.parquet", engine="fastparquet")['pred']  
                df_val_ood_both[X] = pd.read_parquet(val_path + "ood_both.parquet", engine="fastparquet")['pred']    
            except:
                raise Exception()     
    
    if args.augment_offline:
        x = df_train[X].to_list() + df_train[args.augment_offline].to_list()
        y = df_train[Y].to_list() * 2
        train_set = AdsorptionRegressionDataset(x, y, tokenizer, args.max_len, True, args.xval, augment=args.augment_online) 
    else:
        train_set = AdsorptionRegressionDataset(df_train[X], df_train[Y], tokenizer, args.max_len, True, args.xval, augment=args.augment_online) 

    if args.val_sample:
        df_val_id_sample = df_val_id.sample(frac=args.val_sample)
        df_val_ood_ads_sample = df_val_ood_ads.sample(frac=args.val_sample)
        df_val_ood_cat_sample = df_val_ood_cat.sample(frac=args.val_sample)
        df_val_ood_both_sample = df_val_ood_both.sample(frac=args.val_sample)
    else:
        df_val_id_sample = df_val_id
        df_val_ood_ads_sample = df_val_ood_ads
        df_val_ood_cat_sample = df_val_ood_cat
        df_val_ood_both_sample = df_val_ood_both
    
    val_set_id = AdsorptionRegressionDataset(df_val_id_sample[X], df_val_id_sample[Y], tokenizer, args.max_len, train_set.scaler, args.xval)
    val_set_ood_ads = AdsorptionRegressionDataset(df_val_ood_ads_sample[X], df_val_ood_ads_sample[Y], tokenizer, args.max_len, train_set.scaler, args.xval)
    val_set_ood_cat = AdsorptionRegressionDataset(df_val_ood_cat_sample[X], df_val_ood_cat_sample[Y], tokenizer, args.max_len, train_set.scaler, args.xval)
    val_set_ood_both = AdsorptionRegressionDataset(df_val_ood_both_sample[X], df_val_ood_both_sample[Y], tokenizer, args.max_len, train_set.scaler, args.xval)
    val_dict = {'id': val_set_id, 
                'ood_ads': val_set_ood_ads, 
                'ood_cat': val_set_ood_cat, 
                'ood_both': val_set_ood_both}
   
    if args.xval:  # xval is for numerical encoding
        model = NumBertForSequenceClassification.from_pretrained(
            PRETRAINED_MODEL_PATH, num_labels=1)
    else:
        model = BertForSequenceClassification.from_pretrained(
            PRETRAINED_MODEL_PATH, num_labels=1)
    model.dropout = nn.Dropout(0.2)
    model.config.classifier_dropout = 0.2
    model.config.problem_type = "regression"
    model.num_labels = 1
    
    training_args = TrainingArguments(
        report_to= "tensorboard",
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="steps",
        logging_strategy="steps",
        save_strategy="steps",
        logging_first_step=True,
        do_train=True,
        do_eval=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        warmup_steps=1000,
        overwrite_output_dir=True,
        save_total_limit=1,
        save_only_model=False,
        load_best_model_at_end=True,  # For inference!
        metric_for_best_model=args.metric_for_best,
        greater_is_better=False,  
        fp16=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        lr_scheduler_type="linear",
        lr_scheduler_kwargs={},
        eval_on_start=True
    )
        
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_dict,
        compute_metrics=compute_metrics_regression_w(train_set.scaler), 
        compute_loss_func=MAE if args.mae_loss else None,
    )
    
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint) 

    # Save scaler and vocabulary
    save(train_set.scaler, OUTPUT_DIR + "/scaler.pt")  
    shutil.copy(VOCAB, OUTPUT_DIR + "/vocabulary.txt")

    # INFERENCE PIPELINE
    trainer.evaluate()
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.val_sample: # create datasets on entire original validation sets!
        val_set_id = AdsorptionRegressionDataset(df_val_id[X], df_val_id[Y], tokenizer, args.max_len, train_set.scaler, args.xval)
        val_set_ood_ads = AdsorptionRegressionDataset(df_val_ood_ads[X], df_val_ood_ads[Y], tokenizer, args.max_len, train_set.scaler, args.xval)
        val_set_ood_cat = AdsorptionRegressionDataset(df_val_ood_cat[X], df_val_ood_cat[Y], tokenizer, args.max_len, train_set.scaler, args.xval)
        val_set_ood_both = AdsorptionRegressionDataset(df_val_ood_both[X], df_val_ood_both[Y], tokenizer, args.max_len, train_set.scaler, args.xval)
    
    BATCH_SIZE = 256
    loader_train = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)  # shuffle=False to keep order
    loader_id = DataLoader(val_set_id, batch_size=BATCH_SIZE, shuffle=False)
    loader_ood_ads = DataLoader(val_set_ood_ads, batch_size=BATCH_SIZE, shuffle=False)
    loader_ood_cat = DataLoader(val_set_ood_cat, batch_size=BATCH_SIZE, shuffle=False)
    loader_ood_both = DataLoader(val_set_ood_both, batch_size=BATCH_SIZE, shuffle=False)

    train_pred, val_id_pred, val_ood_ads_pred, val_ood_cat_pred, val_ood_both_pred = [], [], [], [], []
    run_inference_regression(loader_train, device, trainer.model, train_pred, train_set.scaler)
    run_inference_regression(loader_id, device, trainer.model, val_id_pred, train_set.scaler)
    run_inference_regression(loader_ood_ads, device, trainer.model, val_ood_ads_pred, train_set.scaler)
    run_inference_regression(loader_ood_cat, device, trainer.model, val_ood_cat_pred, train_set.scaler)
    run_inference_regression(loader_ood_both, device, trainer.model, val_ood_both_pred, train_set.scaler)

    df_train['pred'] = train_pred
    df_train.to_parquet(OUTPUT_DIR + "/train_predictions.parquet")

    df_val_id['pred'] = val_id_pred
    df_val_ood_ads['pred'] = val_ood_ads_pred
    df_val_ood_cat['pred'] = val_ood_cat_pred
    df_val_ood_both['pred'] = val_ood_both_pred
    df_val_id['split'] = 'id'
    df_val_ood_ads['split'] = 'ads'
    df_val_ood_cat['split'] = 'cat'
    df_val_ood_both['split'] = 'both'
    df_val = pd.concat([df_val_id, df_val_ood_ads, df_val_ood_cat, df_val_ood_both])
    df_val.to_parquet(OUTPUT_DIR + "/val_predictions.parquet")

if __name__ == "__main__": 
    main()