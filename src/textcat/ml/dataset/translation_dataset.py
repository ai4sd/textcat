"""
Dataset class for translation task from 
Initial state to Final relaxed adsorption (IS2RS). 
Args must be provided as Pandas dataframe columns.

If xval_encoding is used for numbers, (i) vocab must
contain [NUM] special token, (ii) the tokenizer 
must implement a self.extract_numbers(str)->list[float] function,
and max_length must be large enough to ensure absence of truncation.
"""

from typing import Any, Union

import numpy as np
import pandas as pd
from pytoda.smiles.transforms import Augment
import torch

class AdsorptionTranslationDataset(torch.utils.data.Dataset):
    """
    Dataset for translation task from IS to Relaxed Structure.
    Input: HetSMILES tokens defining the initial state.
    Label: HetSMILES tokens defining the final state.

    Important:
    - Online Data augmentation is possible but only for the x!
    - Possible to duplicate dataset with RS, s.t RS -> RS.
    """
    
    def __init__(self, 
                 x_column: Union[list[str], pd.Series], 
                 y_column: Union[list[str], pd.Series],
                 tokenizer: Any, 
                 max_len: int=512, 
                 xval_encoding: bool=False, 
                 augment: bool=False, 
                 rs2rs: bool=False):
        
        if isinstance(x_column, list):
            self.x = x_column
        else:
            self.x = x_column.to_list()
        if isinstance(y_column, list):
            self.y = y_column if len(y_column) == len(x_column) else [''] * len(x_column)
        else:
            self.y = y_column.to_list() 
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.xval_encoding = xval_encoding
        if self.xval_encoding:
            self.num_token_id = self.tokenizer.convert_tokens_to_ids('[NUM]')
        else:
            self.num_token_id = None
        
        self.augment = augment
        if self.augment:
            self.aug = Augment(kekule_smiles=False, 
                               all_bonds_explicit=False, 
                               all_hs_explicit=True, 
                               sanitize=False)
        else:
            self.aug = None

        self.rs2rs = rs2rs
        if self.rs2rs:
            self.x.extend(self.y)
            self.y.extend(self.y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        smiles = self.x[index]
        if self.augment:
            smiles = self.aug(smiles)
        source_encoding = self.tokenizer(
            smiles,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target_encoding = self.tokenizer(
            self.y[index],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Remove the batch dimension
        source_ids = source_encoding["input_ids"].squeeze()
        source_mask = source_encoding["attention_mask"].squeeze()
        target_ids = target_encoding["input_ids"].squeeze()
        target_ids[target_ids == self.tokenizer.pad_token_id] = -100
        target_mask = target_encoding["attention_mask"].squeeze()

        outputs = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": target_ids,
            "decoder_attention_mask": target_mask,
        }

        if self.xval_encoding:
            x_extracted_nums = self.tokenizer.basic_tokenizer.extract_numbers(smiles)
            y_extracted_nums = self.tokenizer.basic_tokenizer.extract_numbers(self.y[index])
            x_numbers = np.ones_like(source_ids, dtype=float)
            y_numbers = np.ones_like(target_ids, dtype=float)
            x_no_trunc = np.sum(np.array(source_ids)==self.num_token_id)  # useful when max_len is small and truncation occurs
            y_no_trunc = np.sum(np.array(target_ids)==self.num_token_id)  # useful when max_len is small and truncation occurs
            x_numbers[np.where(np.array(source_ids)==self.num_token_id)] = x_extracted_nums[:x_no_trunc]
            y_numbers[np.where(np.array(target_ids)==self.num_token_id)] = y_extracted_nums[:y_no_trunc]
            outputs["input_num"] =  torch.tensor(x_numbers, dtype=torch.float)
            outputs["labels_num"] =  torch.tensor(y_numbers, dtype=torch.float)
            return outputs
        return outputs