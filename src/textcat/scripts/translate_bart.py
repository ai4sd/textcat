"""
Script for training HuggingFace Seq2Seq models (BART/T5) for adsorption
relaxation from initial to relaxed state with OC20 dataset.
Included inference pipeline at the end with best model.

Learning rate = 1e-4 for verybig BART (340M) leads to unstable training, set to 1e-5.
Note:
- args.val sample allows to monitor the metrics on subsets of the original validation sets
  during training. However, the final inference pipeline is run on the whole validation sets.
"""

import argparse
import shutil

import pandas as pd
from rdkit import RDLogger
from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    BertModel
)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from torch.utils.data import DataLoader

from textcat.ml.dataset.translation_dataset import AdsorptionTranslationDataset
from textcat.ml.tokenizer.adsorption_tokenizer import AdsorptionTokenizer
from textcat.ml.tokenizer.bert_tokenizer import GeneralTokenizer
from textcat.ml.train_utils import run_inference_translation, compute_metrics_translation_w, xavier_uniform_init
RDLogger.DisableLog('rdApp.*')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--x", type=str, help="Name of the columns containing the initial state.")
    parser.add_argument("--y", type=str, help="Name of the column containing the final state.")
    parser.add_argument("--vocab_path", type=str, help="Vocabulary .txt file. Must be in directory 'data/vocabularies'.", default='vocabulary.txt')
    parser.add_argument("--output_path", type=str, default="models/translation/")
    parser.add_argument(
        "--train_data_path",
        type=str,
        help="Training dataset in .parquet format",
        default="data/dataframes/train/data.parquet",
    )
    parser.add_argument(
        "--val_data_path",
        default="data/dataframes/",
        type=str,
        help="Directory with validation datasets .parquet.",
    )
    parser.add_argument("--learning_rate", type=float, default=0.00001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--num_epochs", type=float, default=10)
    parser.add_argument("--num_cpus", type=int, default=4)
    parser.add_argument("--eval_steps", type=int, default=25000)
    parser.add_argument("--save_steps", type=int, default=25000)
    parser.add_argument("--logging_steps", type=int, default=5000)
    parser.add_argument("--bert_checkpoint", type=str, default=None)
    parser.add_argument("--xval", action='store_true', default=False, help='Enable number encoding for bond lengths in SMILES. Not available for now.')
    parser.add_argument("--remove_tilde", action='store_true', default=False, help='Remove "~" from SMILES before tokenization.')
    parser.add_argument("--model_type", type=str, default="bart", choices=["bart", "t5"])
    parser.add_argument(
        "--remove_train_anomalies",
        type=int,
        nargs="+",
        default=None,
        choices=[1, 2, 3, 4],
        help="Filter out data with specified anomalies from training dataset",
    )
    parser.add_argument(
        "--remove_val_anomalies",
        type=int,
        nargs="+",
        default=None,
        choices=[1, 2, 3, 4],
        help="Filter out data with specified anomalies from validation datasets.",
    )
    parser.add_argument("--resume_from_checkpoint", action='store_true', default=False)
    parser.add_argument("--eval_accumulation_steps", type=int, default=None, help="Useful to reduce out of memory issues during evaluation.")
    parser.add_argument("--augment_online", action='store_true', default=False, help="Train with enumerated SMILES instead of canonical SMILES only.")
    parser.add_argument("--augment_offline", action='store_true', default=False, help="Double dataset size by adding data where source (guess initial state) == target (optimized state).")
    parser.add_argument("--model_size", type=str, choices=["big", "very_big"], default="big")
    parser.add_argument("--val_sample", type=float, default=None, help="Perform validation on a fraction (0-1) of val_sample entries instead of the entire dataset.")
    parser.add_argument("--xavier_init", action='store_true', default=False, help="Initialize BART params with Xavier initialization strategy.")
    args = parser.parse_args()
    print(args)

    # Define paths for model, files, etc. 
    VOCAB = "data/vocabularies/" + args.vocab_path
    OUTPUT_DIR = args.output_path + args.run_name
    X, Y = args.x, args.y
    DF_COLUMNS = [X, Y, 'anomaly']
    tokenizer = GeneralTokenizer(
        vocab_file=VOCAB,
        basic_tokenizer=AdsorptionTokenizer(args.remove_tilde, args.xval),
        remove_mapping=True,
    )

    # Load Dataframes and filter out anomalies
    df_train = pd.read_parquet(args.train_data_path, engine="fastparquet", columns=DF_COLUMNS)
    df_val_id = pd.read_parquet(args.val_data_path + "val_id/data.parquet", engine="fastparquet", columns=DF_COLUMNS)
    df_val_ood_cat = pd.read_parquet(args.val_data_path + "val_ood_cat/data.parquet", engine="fastparquet", columns=DF_COLUMNS)
    df_val_ood_ads = pd.read_parquet(args.val_data_path + "val_ood_ads/data.parquet", engine="fastparquet", columns=DF_COLUMNS)
    df_val_ood_both = pd.read_parquet(args.val_data_path + "val_ood_both/data.parquet", engine="fastparquet", columns=DF_COLUMNS)

    if args.remove_train_anomalies:
        for anomaly_id in args.remove_train_anomalies:
            df_train = df_train[df_train["anomaly"] != anomaly_id]
            print(f"Removed anomaly {anomaly_id} from data")
    if args.remove_val_anomalies:
        for anomaly_id in args.remove_val_anomalies:     
            df_val_id = df_val_id[df_val_id['anomaly'] != anomaly_id]
            df_val_ood_ads = df_val_ood_ads[df_val_ood_ads['anomaly'] != anomaly_id]
            df_val_ood_cat = df_val_ood_cat[df_val_ood_cat['anomaly'] != anomaly_id]
            df_val_ood_both = df_val_ood_both[df_val_ood_both['anomaly'] != anomaly_id]
            print(f"Removed anomaly {anomaly_id} data from validation sets")

    train_set = AdsorptionTranslationDataset(df_train[X], df_train[Y], tokenizer, args.max_len, xval_encoding=args.xval, augment=args.augment_online, rs2rs=args.augment_offline)     
    if args.val_sample:
        # If provided, create duplicates, one for final inference on all data and one for monitoring training with subset
        df_val_id_sample = df_val_id.sample(frac=args.val_sample)
        df_val_ood_ads_sample = df_val_ood_ads.sample(frac=args.val_sample)
        df_val_ood_cat_sample = df_val_ood_cat.sample(frac=args.val_sample)
        df_val_ood_both_sample = df_val_ood_both.sample(frac=args.val_sample)
    else:
        df_val_id_sample = df_val_id
        df_val_ood_ads_sample = df_val_ood_ads
        df_val_ood_cat_sample = df_val_ood_cat
        df_val_ood_both_sample = df_val_ood_both

    val_set_id = AdsorptionTranslationDataset(df_val_id_sample[X], df_val_id_sample[Y], tokenizer, args.max_len, xval_encoding=args.xval)
    val_set_ood_ads = AdsorptionTranslationDataset(df_val_ood_ads_sample[X], df_val_ood_ads_sample[Y], tokenizer, args.max_len, xval_encoding=args.xval)  
    val_set_ood_cat = AdsorptionTranslationDataset(df_val_ood_cat_sample[X], df_val_ood_cat_sample[Y], tokenizer, args.max_len, xval_encoding=args.xval)  
    val_set_ood_both = AdsorptionTranslationDataset(df_val_ood_both_sample[X], df_val_ood_both_sample[Y], tokenizer, args.max_len, xval_encoding=args.xval)  

    val_dict = {'id': val_set_id, 
                'ood_ads': val_set_ood_ads, 
                'ood_cat': val_set_ood_cat, 
                'ood_both': val_set_ood_both}

    if args.model_type == 'bart':
        config = BartConfig() 
        config.encoder_attention_heads = 8 if args.model_size == 'big' else 16
        config.decoder_attention_heads = 8 if args.model_size == 'big' else 16
        config.decoder_layers = 6 if args.model_size == 'big' else 12
        config.encoder_layers = 6 if args.model_size == 'big' else 12
        config.vocab_size = tokenizer.vocab_size
        config.max_position_embeddings = args.max_len
        config.d_model = 512 if args.model_size == 'big' else 1024
        config.encoder_ffn_dim = 2048 if args.model_size == 'big' else 4096
        config.decoder_ffn_dim = 2048 if args.model_size == 'big' else 4096
        config.dropout = 0.1
        model = BartForConditionalGeneration(config)
        if args.xavier_init:
            model.apply(xavier_uniform_init)
    elif args.model_type == 't5':
        tokenizer = AutoTokenizer.from_pretrained("GT4SD/multitask-text-and-chemistry-t5-small-standard")
        model = AutoModelForSeq2SeqLM.from_pretrained("GT4SD/multitask-text-and-chemistry-t5-small-standard")
        config = {}
    else:
        return Exception
    print(config)
    if args.bert_checkpoint:
        print(f'Loading encoder Bert model from: {args.bert_checkpoint}')
        model.encoder = BertModel.from_pretrained(args.bert_checkpoint)
    print(model.num_parameters())

    data_collator = DataCollatorForSeq2Seq(tokenizer, 
                                           model=model, 
                                           padding='max_length', 
                                           max_length=args.max_len, 
                                           return_tensors='pt')

    training_args = TrainingArguments(
        report_to="tensorboard",
        learning_rate=args.learning_rate,
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="steps",
        logging_strategy="steps",
        logging_first_step=True,
        do_train=True,
        do_eval=True,
        load_best_model_at_end=True,  # For inference!
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        warmup_steps=1000,
        overwrite_output_dir=True,  # careful
        metric_for_best_model="eval_ood_ads_top1_accuracy",
        greater_is_better=True,
        save_total_limit=1,
        fp16=True,
        dataloader_num_workers=args.num_cpus,
        dataloader_pin_memory=True,
        num_train_epochs=args.num_epochs,
        lr_scheduler_type="linear", # linear
        lr_scheduler_kwargs={},
        eval_accumulation_steps=args.eval_accumulation_steps, 
        eval_on_start=True
    )

    print(f"Training args {training_args}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_dict,
        compute_metrics=compute_metrics_translation_w(tokenizer),
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Store vocabulary
    shutil.copy(VOCAB, OUTPUT_DIR + "/vocabulary.txt")

    # INFERENCE PIPELINE (best model)
    trainer.evaluate()
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # check validation sets
    if args.val_sample: # create datasets on entire original validation sets! if not, the ones already created are already ok
        val_set_id = AdsorptionTranslationDataset(df_val_id[X], df_val_id[Y], tokenizer, args.max_len, xval_encoding=args.xval)
        val_set_ood_ads = AdsorptionTranslationDataset(df_val_ood_ads[X], df_val_ood_ads[Y], tokenizer, args.max_len, xval_encoding=args.xval)  
        val_set_ood_cat = AdsorptionTranslationDataset(df_val_ood_cat[X], df_val_ood_cat[Y], tokenizer, args.max_len, xval_encoding=args.xval)  
        val_set_ood_both = AdsorptionTranslationDataset(df_val_ood_both[X], df_val_ood_both[Y], tokenizer, args.max_len, xval_encoding=args.xval)  

    BATCH_SIZE = 32
    loader_train = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)  # shuffle=False to mantain order
    loader_id = DataLoader(val_set_id, batch_size=BATCH_SIZE, shuffle=False)
    loader_ood_ads = DataLoader(val_set_ood_ads, batch_size=BATCH_SIZE, shuffle=False)
    loader_ood_cat = DataLoader(val_set_ood_cat, batch_size=BATCH_SIZE, shuffle=False)
    loader_ood_both = DataLoader(val_set_ood_both, batch_size=BATCH_SIZE, shuffle=False)
    
    train_pred, val_id_pred, val_ood_ads_pred, val_ood_cat_pred, val_ood_both_pred = [], [], [], [], []
    run_inference_translation(loader_train, device, trainer.model, tokenizer, train_pred)
    run_inference_translation(loader_id, device, trainer.model, tokenizer, val_id_pred)
    run_inference_translation(loader_ood_ads, device, trainer.model, tokenizer, val_ood_ads_pred)
    run_inference_translation(loader_ood_cat, device, trainer.model, tokenizer, val_ood_cat_pred)
    run_inference_translation(loader_ood_both, device, trainer.model, tokenizer, val_ood_both_pred)

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
