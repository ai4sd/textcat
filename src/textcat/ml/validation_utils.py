"""
This module contains functions to 
collect and process metrics and predictions 
from multiple trained models at the same.
For both translation and regression.

- Functions assume that the structure of the directory 
with the models and validation predictions is the following:

models/
    translation/
        o1o1/
            model1/
            model2/
            ...
        o2o1/
            model1/
            model2/
            ...
        o2o2/
            model1/
            model2/
            ...
    regression/
        is/     # Initial State to energy
            mlm/    # pretrained models
                model1/
                model2/
                ...
            ft/     # finetuned models
                model1/
                model2/
                ...
        fs/     # Final State to energy
            mlm/
                model1/
                model2/
                ...
            ft/
                model1/
                model2/
                ...
        fs_pred/    # Final State (predicted from translation model) to energy
            mlm/
                ...
            ft/
                ...
"""

from collections import Counter
from io import StringIO
import os
import sys
import re

import pandas as pd
from rdkit import Chem
from sklearn.metrics import r2_score
from transformers import BartForConditionalGeneration, BertForSequenceClassification
from torch import load

from textcat.ml.tokenizer.bert_tokenizer import GeneralTokenizer
from textcat.ml.tokenizer.adsorption_tokenizer import AdsorptionTokenizer
from textcat.ml.train_utils import get_last_checkpoint


def read_translation_val_predictions_from_subdirectories(dfx: pd.DataFrame, 
                                             directory: str, 
                                             vocab_path: str
                                             ) -> tuple[pd.DataFrame, dict]:
    """
    Given the path containing trained translation models grouped 
    by type oxoy (x,y={1,2}), collect all the predictions on the 
    validation sets and model hyperparameters.

    To work properly, model directories must contain:
    (i) vocabulary.txt, (ii) val_predictions.parquet all
    of the same size to then be concatenated in a single
    Pandas Dataframe.

    2 baseline models are also included: (i) copy model y=x, 
    and (ii) y = most frequent token repeated max_len times.
    """

    hyperparams_dict = {}

    # 1) ADD BASELINE WHERE RS == IS (model just copies the IS)
    name = 'Baseline_copy_input'
    settings = {}
    settings['order'] = 'o1o1'
    settings['X'] = 'hetsmiles_IS_o1'
    settings['Y'] = 'hetsmiles_FS_o1'
    settings['size_M'] = 0
    settings['aug_on'] = False
    settings["aug_off"] = False
    settings['notilde'] = True
    settings["twall"] = True
    settings['tokenizer'] = GeneralTokenizer(vocab_path, 
                                            AdsorptionTokenizer(True, False))
    settings["max_len"] = 176
    settings['path'] = 'N/A'
    hyperparams_dict[name] = settings
    predictions = dfx[settings['X']].to_list()
    predictions = [y.replace("~", "") for y in predictions]    
    df = pd.concat([dfx, pd.DataFrame(predictions, columns=['Y'+name])], axis=1)
    print('Added baseline copy model (predicted relaxed state coincides with input initial state)')

    # 2) ADD BASELINE WHERE RS == repetition of most frequent token
    name = 'Baseline_most_frequent_token'
    settings = {}
    settings['order'] = 'o1o1'
    settings['X'] = 'hetsmiles_IS_o1'
    settings['Y'] = 'hetsmiles_FS_o1'
    settings['size_M'] = 0
    settings['aug_on'] = False
    settings["aug_off"] = False
    settings['notilde'] = True
    settings["twall"] = True
    settings['tokenizer'] = GeneralTokenizer(vocab_path, 
                                            AdsorptionTokenizer(True, False))
    settings["max_len"] = 176 
    settings['path'] = 'N/A'
    hyperparams_dict[name] = settings
    train_tokens = [settings['tokenizer'].tokenize(x.replace("~", "")) for x in dfx[settings['X']].to_list()]
    all_tokens = [token for sublist in train_tokens for token in sublist]
    token_counts = Counter(all_tokens)
    most_frequent_token, most_frequent_count = token_counts.most_common(1)[0]
    predictions = [most_frequent_token * settings["max_len"] for _ in range(len(dfx))]
    df = pd.concat([df, pd.DataFrame(predictions, columns=['Y'+name])], axis=1)
    print(f'Added baseline most frequent token model ("{most_frequent_token}" token appearing {most_frequent_count} times)')

    counter = 0
    for tr_type in os.listdir(directory):
        print(f"Processing {tr_type} models")
        tr_type_path = os.path.join(directory, tr_type)
        for model_name in os.listdir(tr_type_path):
            try:
                model_path = os.path.join(tr_type_path, model_name)
                settings = {}
                settings['order'] = tr_type
                settings['X'] = f'hetsmiles_IS_{tr_type[:2]}'
                settings['Y'] = f'hetsmiles_FS_{tr_type[2:]}'
                model = BartForConditionalGeneration.from_pretrained(get_last_checkpoint(model_path))
                settings['size_M'] = round(model.num_parameters()/1e6, 1)
                settings['aug_on'] = True if "on" in model_name else False
                settings["aug_off"] = True if "off" in model_name else False
                settings["twall"] = True if "twall" in model_name else False
                settings["max_len"] = 176 if settings['order'] == 'o1o1' else 512
                settings['notilde'] = True if "wotilde" in model_name else False
                settings['tokenizer'] = GeneralTokenizer(model_path + '/vocabulary.txt', 
                                    AdsorptionTokenizer(settings['notilde'], False))
                settings['path'] = os.path.abspath(model_path)
                hyperparams_dict[str(counter+1)] = settings
                
                if os.path.isdir(model_path):
                    parquet_path = os.path.join(model_path, 'val_predictions.parquet')
                    if os.path.exists(parquet_path):
                        predictions = pd.read_parquet(parquet_path, columns=['pred'])['pred'].to_list()               
                        df = pd.concat([df, pd.DataFrame(predictions, columns=[f'Y{counter+1}'])], 
                                    axis=1)
                        counter += 1   
                    elif 'val_id_pred.parquet' in os.listdir(model_path):
                        df_id = pd.read_parquet(os.path.join(model_path, 'val_id_pred.parquet'))
                        df_ads = pd.read_parquet(os.path.join(model_path, 'val_ood_ads_pred.parquet'))
                        df_cat = pd.read_parquet(os.path.join(model_path, 'val_ood_cat_pred.parquet'))
                        df_both = pd.read_parquet(os.path.join(model_path, 'val_ood_both_pred.parquet'))
                        dfv = pd.concat([df_id, df_ads, df_cat, df_both], axis=0, ignore_index=True)
                        predictions = dfv['pred'].to_list()
                        df = pd.concat([df, pd.DataFrame(predictions, columns=[f'Y{counter+1}'])], 
                                    axis=1)
                        counter += 1
                    else:
                        continue
            except:
                continue
    print(f'(Found {counter} adsorption translation models)')
    return df, hyperparams_dict


def extract_substrings_in_brackets(input_string):
    substrings = re.findall(r'\[([^\]]+)\]', input_string)
    return set(substrings)


def generate_translation_metrics(df: pd.DataFrame,  
                                exps: dict[int, dict]) -> None:
    """
    Collect metrics from provided experiments
    on validation data.
    df: each row is a data entry
    exps: each item is an experiment/trained model

    Collected metrics are added to exps.

    For each predicted SMILES and its true value, the 
    following metrics are collected: 
    - sequence equality (bool, character-wise)
    - presence of all elements in predicted SMILES (bool)
    - SMILES validity (bool)
    - RDKit error (str, if SMILES unvalid)
    - Length difference between true and pred SMILES (int, char.-wise)
    - Token accuracy sequence-wise (float, %)

    For each experiment, the following overall 
    metrics are collected:
    - Average SMILES validity (%)
    - Average token accuracy per SMILES (%)
    - Average sequence accuracy (%)
    - Average SMILES length difference (char-wise) (float)
    The four metrics are computed (i) globally, (ii) by validation 
    split (id, ood_ads, ood_cat, ood_both), (iii)  globally without 
    data of anomaly 1/4, (iv) by split without data of anomaly 1/4.
    """
    
    print(f"Generating metrics for {len(exps)} models")
    sio = sys.stderr = StringIO()

    for IDX in exps.keys():
        print(f"Experiment {IDX}")
        smiles_validity_list, equal_list, smiles_error_msg_list, all_elems_present, smiles_length_err = [], [], [], [], []        
        tokens_correct_pctg = [] 
        for true, pred in zip(df[exps[IDX]['Y']], df[f'Y{IDX}']):
            y = true.replace("~", "") if exps[IDX]['notilde'] else true
            equal_list.append(pred == y)
            elems_true = extract_substrings_in_brackets(true)
            elems_pred = extract_substrings_in_brackets(pred)
            all_elems_present.append(True if elems_true == elems_pred else False)
            mol_pred = Chem.MolFromSmiles(pred, sanitize=False)
            valid_smiles =  True if isinstance(mol_pred, Chem.rdchem.Mol) else False
            smiles_validity_list.append(valid_smiles)
            error_value = sio.getvalue()
            if error_value != '':
                try:
                    x = error_value.split(':')[3]
                    xx = ''
                    for char in x:
                        xx += char if not char.isnumeric() else ''
                    smiles_error_msg_list.append(xx)
                except:
                    smiles_error_msg_list.append('N/A')
            else:
                smiles_error_msg_list.append('Valid')
            sio = sys.stderr = StringIO() # reset
            smiles_length_err.append(len(y) - len(pred))
            tokens_true = exps[IDX]["tokenizer"].tokenize(true)
            tokens_pred = exps[IDX]["tokenizer"].tokenize(pred)
            tokens_correct_pctg.append(round(sum([i == j for i, j in zip(tokens_true, tokens_pred)]) / len(tokens_true) * 100.0, 2))

        df[f'valid_smiles{IDX}'] = smiles_validity_list
        df[f'equal{IDX}'] = equal_list
        df[f'error{IDX}'] = smiles_error_msg_list
        df[f'all_elems{IDX}'] = all_elems_present
        df[f'len_err{IDX}'] = smiles_length_err
        df[f'correct_toks{IDX}'] = tokens_correct_pctg

        # METRICS EXP-WISE (global/by split, with/without anomaly data)
        # Global metrics
        exps[IDX]["smiles_val"] = round(df[f'valid_smiles{IDX}'].mean() * 100, 2)
        exps[IDX]["tok_acc"] = round(df[f'correct_toks{IDX}'].mean(), 2)
        exps[IDX]["seq_acc"] = round(df[f'equal{IDX}'].mean() * 100, 2)
        exps[IDX]["len_acc"] = round(df[f'len_err{IDX}'].mean(), 2)
        # Metrics by validation split
        for split, group in df.groupby('split'):
            exps[IDX][f"smiles_val_{split}"] = round(group[f'valid_smiles{IDX}'].mean() * 100, 2)
            exps[IDX][f"tok_acc_{split}"] = round(group[f'correct_toks{IDX}'].mean(), 2)
            exps[IDX][f"seq_acc_{split}"] = round(group[f'equal{IDX}'].mean() * 100, 2)
            exps[IDX][f"len_acc{split}"] = round(group[f'len_err{IDX}'].mean(), 2)
        # Global metrics without data with anomaly 1 or 4
        filtered_df = df[~df['anomaly'].isin([1, 4])]
        exps[IDX]["smiles_val_wo14"] = round(filtered_df[f'valid_smiles{IDX}'].mean() * 100, 2)
        exps[IDX]["tok_acc_wo14"] = round(filtered_df[f'correct_toks{IDX}'].mean(), 2)
        exps[IDX]["seq_acc_wo14"] = round(filtered_df[f'equal{IDX}'].mean() * 100, 2)
        exps[IDX]["len_acc_wo14"] = round(filtered_df[f'len_err{IDX}'].mean(), 2)
        # Metrics by validation split and without data with anomaly 1 and 4
        for split, group in filtered_df.groupby('split'):
            exps[IDX][f"smiles_val_wo14_{split}"] = round(group[f'valid_smiles{IDX}'].mean() * 100, 2)
            exps[IDX][f"tok_acc_wo14_{split}"] = round(group[f'correct_toks{IDX}'].mean(), 2)
            exps[IDX][f"seq_acc_wo14_{split}"] = round(group[f'equal{IDX}'].mean() * 100, 2)
            exps[IDX][f"len_acc_wo14_{split}"] = round(group[f'len_err{IDX}'].mean(), 2)


def read_regression_val_predictions_from_subdirectories(df: pd.DataFrame, 
                                             directory: str) -> tuple[pd.DataFrame, dict]:
    """
    Given the path containing trained regression models, 
    collect all the predictions on the validation sets 
    and model hyperparameters.

    To work properly, model directories must contain:
    (i) vocabulary.txt, (ii) val_predictions.parquet all
    of the same size to then be concatenated into a single
    Pandas Dataframe, (iii) target scaler.pt used during training.

    Input directory is the root models/regression/
    """
    hyperparams_dict = {}    
    counter = 0
    strategy_dict = {'is/ft':'IS2RE_direct', 'fs/ft':'RS2RE', 'fs_pred/ft': 'IS2RE_translation'}
    for regression_type in ['is/ft', 'fs/ft', 'fs_pred/ft']:  # ft=finetuning
        type_path = os.path.join(directory, regression_type)
        print(f"Processing {strategy_dict[regression_type]} to relaxed energy models...")
        for model_name in os.listdir(type_path):
            model_path = os.path.join(type_path, model_name)
            settings = {}
            x = 'FS' if '_fs_' in model_name else 'IS'
            order = 'o1' if '_o1_' in model_path else "o2"
            settings['order'] = order
            settings['X'] = f'hetsmiles_{x}_{order}'
            settings['Y'] = 'eads_eV'
            settings['strategy'] = strategy_dict[regression_type]
            try:  # Just skip in case something (model, predictions) is missing in the folder!
                model = BertForSequenceClassification.from_pretrained(get_last_checkpoint(model_path))
                settings['size_M'] = round(model.num_parameters()/1e6, 1)
                settings['aug_on'] = True if "on" in model_name else False
                settings["aug_off"] = True if "off" in model_name else False
                settings["twall"] = True if "twall" in model_name else False
                settings["max_len"] = 176 if settings['order'] == 'o1' else 512
                settings['notilde'] = True if "wotilde" in model_name else False
                settings['tokenizer'] = GeneralTokenizer(model_path + '/vocabulary.txt', 
                                    AdsorptionTokenizer(settings['notilde'], False))
                settings['y_scaler'] = load(model_path + '/scaler.pt')
                settings['path'] = os.path.abspath(model_path)
                hyperparams_dict[str(counter+1)] = settings            
                if os.path.isdir(model_path):
                    parquet_path = os.path.join(model_path, 'val_predictions.parquet')
                    if os.path.exists(parquet_path):
                        predictions = pd.read_parquet(parquet_path, columns=['pred'])['pred'].to_list()               
                        df[f'Y{counter+1}'] = predictions
                        counter += 1                        
                    else: # In case validation predictions are split in 4 different files
                        df_id = pd.read_parquet(os.path.join(model_path, 'val_id_pred.parquet'))
                        df_ads = pd.read_parquet(os.path.join(model_path, 'val_ood_ads_pred.parquet'))
                        df_cat = pd.read_parquet(os.path.join(model_path, 'val_ood_cat_pred.parquet'))
                        df_both = pd.read_parquet(os.path.join(model_path, 'val_ood_both_pred.parquet'))
                        dfv = pd.concat([df_id, df_ads, df_cat, df_both], axis=0, ignore_index=True)
                        predictions = dfv['pred'].to_list()
                        df[f'Y{counter+1}'] = predictions
                        counter += 1
            except:
                continue

    # ADD BASELINE WHERE eads_eV_pred == mean of training data (both with and without anomalies 1 and 4)
    for case in [True, False]:
        settings = {}
        settings['order'] = 'mean'
        settings['X'] = 'N/A'
        settings['Y'] = 'eads_eV'
        settings['size_M'] = 0
        settings['aug_on'] = False
        settings["aug_off"] = False
        settings['notilde'] = True
        settings["twall"] = case
        for k, v in hyperparams_dict.items():
            if v['twall'] == case:            
                settings['tokenizer'] = hyperparams_dict[k]['tokenizer']
                settings['y_scaler'] = hyperparams_dict[k]['y_scaler']
                break
        settings["max_len"] = 176 if settings['order'] == 'o1o1' else 512
        settings['path'] = 'N/A'
        settings['strategy'] = 'TRAINING_SET_MEAN'
        hyperparams_dict[str(counter+1)] = settings  # BM: Baseline Mean
        df[f'Y{counter+1}'] = settings['y_scaler'].mean_[0]  # Mean is from the training data
        counter += 1

    print(f'(Found {counter - 2} adsorption regression models)')
    return df, hyperparams_dict


def generate_regression_metrics(df: pd.DataFrame,  
                                exps: dict[int, dict]) -> None:
    """
    Collect metrics from provided regression 
    experiments on validation data.
    df: each row is a data entry
    exps: each item is an experiment/trained model

    Collected metrics are added to exps.

    For each predicted energy and its true value, the 
    following metrics are collected: 
    - Error (float, in eV)
    - Absolute error (float, in eV)

    For each experiment, the following metrics are collected:
    - Mean Absolute Error (MAE, in eV)
    - R2 score (float)
    - Root Mean Squared Error (RMSE, in eV)
    - Median Absolute Error (MDAE, in eV)
    The four metrics are computed (i) globally, (ii) by validation 
    split (id, ood_ads, ood_cat, ood_both), (iii)  globally without 
    data of anomaly 1/4, (iv) by split without data of anomaly 1/4.
    """
    
    print(f"Generating metrics for {len(exps)} regression models")

    for IDX in exps.keys():
        print(f"Experiment {IDX} ({exps[IDX]['strategy']})")
        df[f'err{IDX}'] = df[exps[IDX]['Y']] - df[f'Y{IDX}']
        df[f'abs_err{IDX}'] = df[f'err{IDX}'].abs()
        
        # METRICS EXP-WISE (global/by split, with/without anomaly data)
        # 1) Global metrics
        exps[IDX]["mae"] = round(df[f'abs_err{IDX}'].mean(), 3)
        exps[IDX]["rmse"] = round(df[f'err{IDX}'].pow(2).mean() ** 0.5, 3)
        exps[IDX]["r2"] = round(r2_score(df[exps[IDX]['Y']], df[f'Y{IDX}']), 3)
        exps[IDX]["mdae"] = round(df[f'abs_err{IDX}'].median(), 3)
        # 2) Metrics by validation split
        for split, group in df.groupby('split'):
            exps[IDX][f"mae_{split}"] = round(group[f'abs_err{IDX}'].mean(), 3)
            exps[IDX][f"rmse_{split}"] = round(group[f'err{IDX}'].pow(2).mean() ** 0.5, 3)
            exps[IDX][f"r2_{split}"] = round(r2_score(group[exps[IDX]['Y']], group[f'Y{IDX}']), 3)
            exps[IDX][f"mdae{split}"] = round(group[f'abs_err{IDX}'].median(), 3)
        # 3) Global metrics without data with anomaly 1 and 4
        filtered_df = df[~df['anomaly'].isin([1, 4])]
        exps[IDX]["mae_wo14"] = round(filtered_df[f'abs_err{IDX}'].mean(), 3)
        exps[IDX]["rmse_wo14"] = round(filtered_df[f'err{IDX}'].pow(2).mean() ** 0.5, 3)
        exps[IDX]["r2_wo14"] = round(r2_score(filtered_df[exps[IDX]['Y']], filtered_df[f'Y{IDX}']), 3)
        exps[IDX]["mdae_wo14"] = round(filtered_df[f'abs_err{IDX}'].median(), 3)
        # 4) Metrics by validation split withtou data with anomaly 1 and 4
        for split, group in filtered_df.groupby('split'):
            exps[IDX][f"mae_wo14_{split}"] = round(group[f'abs_err{IDX}'].mean(), 3)
            exps[IDX][f"rmse_wo14_{split}"] = round(group[f'err{IDX}'].pow(2).mean() ** 0.5, 3)
            exps[IDX][f"r2_wo14_{split}"] = round(r2_score(group[exps[IDX]['Y']], group[f'Y{IDX}']), 3)
            exps[IDX][f"mdae_wo14_{split}"] = round(group[f'abs_err{IDX}'].median(), 3)