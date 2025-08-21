"""
Script for pretraining with masked language
modeling (MLM) HuggingFace BERT models.
Training set is OC20.
"""

import argparse

import pandas as pd
from transformers import (
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

from textcat.ml.numbert_wrapper import NumBertForMaskedLM
from textcat.ml.datacollector.num_datacollator import define_masked_num_collator
from textcat.ml.dataset.mlm_dataset import AdsorptionMLMDataset
from textcat.ml.tokenizer.adsorption_tokenizer import AdsorptionTokenizer
from textcat.ml.tokenizer.bert_tokenizer import GeneralTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--x", type=str, help="Name of the column in dataframe.")
    parser.add_argument("--vocab_path", type=str, help="Vocabulary .txt file. Must be in directory 'data/vocabularies'.", default='vocabulary.txt')
    parser.add_argument("--output_path", type=str, default="models/regression/")
    parser.add_argument("--train_data_path", type=str,
                        help="data directory to read train parquet files from",
                        default="data/dataframes/train/data.parquet"
                        )
    parser.add_argument(
        "--val_data_path", default="data/dataframes/",
        type=str,
        help="data directory to read val data parquet file from",
    )
    parser.add_argument("--xval", action='store_true', default=False)
    parser.add_argument("--remove_tilde", action='store_true', default=False)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_epochs", type=int, default=25)
    parser.add_argument("--num_cpus", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=176)
    parser.add_argument("--eval_steps", type=int, default=10000)
    parser.add_argument("--save_steps", type=int, default=10000)
    parser.add_argument("--logging_steps", type=int, default=10000)
    parser.add_argument("--model_size", type=str,
                        default='big', choices=['small', 'big', 'very_big'])
    parser.add_argument("--remove_train_anomalies", type=int, nargs='+', default=None,
                        choices=[1, 2, 3, 4], 
                        help="Filter out data with specified anomalies from training dataset.")
    parser.add_argument("--remove_val_anomalies", type=int, nargs='+', default=None,
                        choices=[1, 2, 3, 4], 
                        help="Filter out data with specified anomalies from validation datasets.")
    parser.add_argument("--augment_online", action='store_true', default=False, help='Augment with enumerated SMILES.')
    parser.add_argument("--augment_offline", type=str, default=None, help='Provide column name to double dataset size by adding cases when IS==RS. Helpful only for IS2RE direct.')
    parser.add_argument("--val_sample", type=float, default=None, help="Perform validation on a fraction (0-1) of val_sample entries instead of the entire dataset.")
    parser.add_argument("--resume_from_checkpoint", action='store_true', default=False)

    args = parser.parse_args()
    print(args)
    
    # Define paths for model, files, etc. 
    VOCAB = "data/vocabularies/" + args.vocab_path
    OUTPUT_DIR = args.output_path + args.run_name
    X = args.x
    DF_COLUMNS = [X, 'anomaly']
    if args.augment_offline:
        DF_COLUMNS.append(args.augment_offline)

    # Load Dataframes and filter out anomalies
    df_train = pd.read_parquet(args.train_data_path, engine="fastparquet", columns=DF_COLUMNS)
    df_val_id = pd.read_parquet(args.val_data_path + "val_id/data.parquet", engine="fastparquet", columns=DF_COLUMNS)
    df_val_ood_cat = pd.read_parquet(args.val_data_path + "val_ood_cat/data.parquet", engine="fastparquet", columns=DF_COLUMNS)
    df_val_ood_ads = pd.read_parquet(args.val_data_path + "val_ood_ads/data.parquet", engine="fastparquet", columns=DF_COLUMNS)
    df_val_ood_both = pd.read_parquet(args.val_data_path + "val_ood_both/data.parquet", engine="fastparquet", columns=DF_COLUMNS)
    if args.remove_train_anomalies:
        for anomaly_id in args.remove_train_anomalies:
            df_train = df_train[df_train['anomaly'] != anomaly_id]
            print(f"Removed anomaly {anomaly_id} from training dataset.")
    if args.remove_val_anomalies:
        for anomaly_id in args.remove_val_anomalies:
            df_val_id = df_val_id[df_val_id['anomaly'] != anomaly_id]
            print(f"Removed anomaly {anomaly_id} from validation dataset.")
    if args.val_sample:
        df_val_id = df_val_id.sample(frac=args.val_sample)
        df_val_ood_ads = df_val_ood_ads.sample(frac=args.val_sample)
        df_val_ood_cat = df_val_ood_cat.sample(frac=args.val_sample)
        df_val_ood_both = df_val_ood_both.sample(frac=args.val_sample)
    
    tokenizer = GeneralTokenizer(
        vocab_file=VOCAB,
        basic_tokenizer=AdsorptionTokenizer(args.remove_tilde, args.xval),
        remove_mapping=True,
    )

    # Instantiate model
    config = BertConfig()
    config.max_position_embeddings = args.max_len
    config.vocab_size = tokenizer.vocab_size
    if args.model_size == 'small':
        config.intermediate_size = 256
        config.hidden_size = 128
        config.num_attention_heads = 2
        config.num_hidden_layers = 6
    elif args.model_size == 'big':
        config.intermediate_size = 512
        config.hidden_size = 256
        config.num_attention_heads = 4
        config.num_hidden_layers = 12
    else:
        config.intermediate_size = 1024
        config.hidden_size = 512
        config.num_attention_heads = 8
        config.num_hidden_layers = 24
    config.hidden_dropout_prob = args.dropout
    if args.xval:
        config.num_token_id = tokenizer.convert_tokens_to_ids('[NUM]')
        model = NumBertForMaskedLM(config)
        data_collator = define_masked_num_collator(
            tokenizer=tokenizer,
            mlm_probability=args.mlm_probability)
    else:
        model = BertForMaskedLM(config)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=args.mlm_probability,
        )
    print("Num params", model.num_parameters())
    if args.augment_offline:
        x = df_train[X].to_list() + df_train[args.augment_offline].to_list()
        train_set = AdsorptionMLMDataset(x, tokenizer, args.max_len, args.xval, args.augment_online)
    else:
        train_set = AdsorptionMLMDataset(df_train[X], tokenizer, args.max_len, args.xval, args.augment_online)
    val_set_id = AdsorptionMLMDataset(df_val_id[X], tokenizer, args.max_len, args.xval, args.augment_online)
    val_set_ood_ads = AdsorptionMLMDataset(df_val_ood_ads[X], tokenizer, args.max_len, args.xval, args.augment_online)
    val_set_ood_cat = AdsorptionMLMDataset(df_val_ood_cat[X], tokenizer, args.max_len, args.xval, args.augment_online)
    val_set_ood_both = AdsorptionMLMDataset(df_val_ood_both[X], tokenizer, args.max_len, args.xval, args.augment_online)

    print(len(train_set))

    val_dict = {'id': val_set_id, 
                'ood_ads': val_set_ood_ads, 
                'ood_cat': val_set_ood_cat, 
                'ood_both': val_set_ood_both}
    
    training_args = TrainingArguments(
        report_to="tensorboard",
        run_name=args.run_name,
        learning_rate=args.learning_rate,
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        evaluation_strategy="steps",
        logging_strategy="steps",
        logging_first_step=True,
        do_train=True,
        do_eval=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        warmup_steps=10000,
        overwrite_output_dir=True,  # careful
        save_total_limit=1,  
        save_only_model=False,
        fp16=True,
        dataloader_num_workers=args.num_cpus,
        dataloader_pin_memory=True,
        load_best_model_at_end=True,
        metric_for_best_model='eval_ood_ads_loss', 
        eval_on_start=True
    )

    print(f"Training args {training_args}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_dict,
        data_collator=data_collator
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

if __name__ == "__main__":
    main()
