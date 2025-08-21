"""
Module containing utility functions for training scripts,
mostly evaluation pipeline and metrics computation. 
"""
import os
import re

import numpy as np
from rdkit import Chem
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm
import torch
from transformers import BertForSequenceClassification

def MAE(outputs, labels, num_items_in_batch=None):
    logits = outputs.get("logits")
    mae_fct = nn.L1Loss()
    return mae_fct(logits, labels)

def NLL(outputs, labels, num_items_in_batch=None):
    logits = outputs.get("logits")
    mae_fct = nn.L1Loss()
    return mae_fct(logits, labels)


def run_inference_regression(loader, 
                             device, 
                             model, 
                             pred_list, 
                             scaler) -> None:
    """
    Run inference for a given data loader and collect the predictions.
    
    Args:
        loader: DataLoader for the current dataset (train, validation, etc.).
        device: The device (CPU/GPU) to run the model on.
        model: The trained model to make predictions.
        tokenizer: The tokenizer to decode the predictions.
        pred_list: The list to store the decoded predictions.
    """
    with torch.no_grad():
        for batch in tqdm.tqdm(loader):
            x = {key: value.to(device) for key, value in batch.items()}
            y = model(**x)['logits'].cpu().numpy()
            y_rescaled = scaler.inverse_transform(y)
            pred_list.extend([x.item() for x in y_rescaled])


def compute_metrics_regression(eval_pred, scaler) -> dict[str, float]:
    y_pred, y_true = eval_pred 
    if scaler:
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
        y_true = scaler.inverse_transform(y_true.reshape(-1, 1))

    return {'RMSE_eV': mean_squared_error(y_true, y_pred, squared=False), 
            'R2': r2_score(y_true, y_pred), 
            'MAE_eV': mean_absolute_error(y_true, y_pred),
            'MDAE_eV': median_absolute_error(y_true, y_pred)}


def compute_metrics_regression_w(scaler):
    """
    Wrapper to the actual_compute metrics function.
    """
    def wrapper(p):
        return compute_metrics_regression(p, scaler)
    return wrapper


def run_inference_translation(loader, 
                              device: str, 
                              model, 
                              tokenizer, 
                              pred_list, 
                              autoregressive=False) -> None:
    """
    Run inference for a given data loader and collect the predictions.
    
    Args:
        loader: DataLoader for the current dataset (train, validation, etc.).
        device: The device (CPU/GPU) to run the model on.
        model: The trained model to make predictions.
        tokenizer: The tokenizer to decode the predictions.
        pred_list: The list to store the decoded predictions.
    """
    with torch.no_grad():
        for batch in tqdm.tqdm(loader):
            x = {key: value.to(device) for key, value in batch.items()}
            if autoregressive:
                y = model.generate()  # To implement properly
            else:
                y = model(**x)['logits'].cpu()  # Get model logits
            pred_ids = torch.argmax(y, dim=-1)  # Get predicted token ids
            decoded_preds = tokenizer.batch_decode(pred_ids, 
                                                skip_special_tokens=True, 
                                                clean_up_tokenization_spaces=True)
            decoded_preds = [y.replace(" ", "").strip() for y in decoded_preds]
            pred_list.extend(decoded_preds)


def compute_metrics_translation(eval_preds, tokenizer) -> dict[str, float]:
    """Returns:
     - token accuracy  
     - top-1 accuracy
     - rdkit-validity
     - Sequence length accuracy"""
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    # Predictions
    probabilities = F.softmax(torch.tensor(preds), dim=-1)
    pred_ids = torch.argmax(probabilities, dim=-1)
    decoded_preds = tokenizer.batch_decode(pred_ids, 
                                           skip_special_tokens=True, 
                                           clean_up_tokenization_spaces=False)
    # Labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, 
                                            skip_special_tokens=True, 
                                            clean_up_tokenization_spaces=False)
    total_tokens = 0
    correct_tokens = 0
    total_top1 = 0
    total_valid= 0
    lengths_diff_sum = 0

    for pred, true in zip(decoded_preds, decoded_labels): # pred  and true are strings, each token spaced, no special ones!
        pred_split = pred.split()
        true_split = true.split()

        total_tokens += len(true_split)
        correct_tokens += sum(p == r for p, r in zip(pred_split, true_split))
        # pred_split and true split do not need to have same length to be zipped!

        total_top1 += 1 if pred == true else 0
        smiles_pred = pred.replace(" ", "")
        smiles_true = true.replace(" ", "")
        total_valid += 1 if isinstance(Chem.MolFromSmiles(smiles_pred, sanitize=False), Chem.Mol) else 0
        lengths_diff_sum += len(smiles_pred) - len(smiles_true)

    # Calculate token accuracy, sequence accuracy and SMILES validity 
    token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    top1_accuracy = total_top1 / len(decoded_preds)
    validity_acc = total_valid / len(decoded_preds)
    mean_length_error = lengths_diff_sum / len(decoded_preds)

    return {
        "token_accuracy": round(token_accuracy, 4),
        "top1_accuracy": round(top1_accuracy, 4), 
        "SMILES_validity": round(validity_acc, 4),
        "mean_length_error": round(mean_length_error, 4)
    }


def compute_metrics_translation_w(tokenizer):
    """
    Wrapper to the actual_compute metrics function.
    """
    def wrapper(p):
        return compute_metrics_translation(p, tokenizer)
    return wrapper


def get_fingerprints(loader: DataLoader, 
                     device: str, 
                     model: BertForSequenceClassification) -> np.ndarray:
    """
    Generate BERT fingerprints (i.e., pooler outputs)
    for a given dataloader.
    """
    fingerprints = []
    for batch in tqdm.tqdm(loader):
        x = {key: value.to(device) for key, value in batch.items() if key != 'labels'}
        y = model.bert(**x)["pooler_output"]
        fingerprints.append(y.cpu().numpy())
    return np.vstack(fingerprints).tolist()


def get_last_checkpoint(x: str) -> str:
    """
    Given a path containing model checkpoints, 
    return the last one. Useful when multiple checkpoints are stored.
    """
    checkpoint_folders = [f for f in os.listdir(x) if os.path.isdir(os.path.join(x, f))]
    step_pattern = re.compile(r"checkpoint-(\d+)")
    return x + "/" + max(checkpoint_folders, key=lambda folder: int(step_pattern.search(folder).group(1)) if step_pattern.search(folder) else -1)


def xavier_uniform_init(m):
    """
    Apply Xavier uniform initialization to model weights.
    This function is applied to all layers in the model.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
