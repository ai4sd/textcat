"""
Torch Dataset class for sequence regression task. 
Args must be provided as Pandas dataframe columns.

If xval_encoding is used for numbers, (i) vocab must
contain [NUM] special token, (ii) the tokenizer 
must implement a self.extract_numbers(str)->list[float] function,
and max_length must be large enough to ensure absence of truncation.
"""

from typing import Union

import numpy as np
import pandas as pd
from pytoda.smiles.transforms import Augment
import torch

from transformers.models.bert import BertTokenizer


class AdsorptionMLMDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 x_column: Union[pd.Series, list[str]], 
                 tokenizer: BertTokenizer, 
                 max_len: int, 
                 xval_encoding: bool=False, 
                 augment: bool=False):      

        """
        Adsorption text dataset for OC20 data.
        """  
        if isinstance(x_column, list):
            self.x = x_column
        else:
            self.x = x_column.to_list()
        
        self.tokenizer = tokenizer        
        self.max_len = max_len
        self.augment = augment
        if self.augment:
            self.aug = Augment(kekule_smiles=False, 
                               all_bonds_explicit=False, 
                               all_hs_explicit=True, 
                               sanitize=False)
        else:
            self.aug = None
        self.xval_encoding = xval_encoding
        if self.xval_encoding:
            self.num_token_id = self.tokenizer.convert_tokens_to_ids('[NUM]')
        else:
            self.num_token_id = None

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index: int) -> dict:
        smiles = self.x[index]
        if self.augment:
            smiles = self.aug(smiles)
        inputs = self.tokenizer(
            smiles,
            None,
            add_special_tokens=True,
            padding='max_length',
            max_length=self.max_len,
            return_token_type_ids=True,
            truncation=True,
        )

        outputs =  {
            "input_ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(inputs["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(inputs["token_type_ids"], dtype=torch.long), 
            "labels" :torch.tensor(inputs["input_ids"], dtype=torch.long),
        }

        if self.xval_encoding:
            extracted_num = self.tokenizer.basic_tokenizer.extract_numbers(self.x[index])
            input_ids = inputs['input_ids']
            numbers = np.ones_like(input_ids, dtype=float)
            no_trunc = np.sum(np.array(input_ids)==self.num_token_id)  # useful when max_len is small and truncation occurs
            numbers[np.where(np.array(input_ids)==self.num_token_id)] = extracted_num[:no_trunc]
            outputs["input_num"] = torch.tensor(numbers, dtype=torch.float)
            outputs["labels_num"] = torch.tensor(numbers, dtype=torch.float)
            return outputs
        return outputs