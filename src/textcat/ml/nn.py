"""
Wrapper class for inferring the adsorption energy from SMILES.

Notes:
- Generated SMILES are RDkit readable with Chem.MolFromSmiles(smiles, sanitize=False)
- Predicted adsorption energy values are in eV and referenced in the same way as the 
data from OC20.
"""

from typing import Union

from rdkit import Chem
from transformers import BertForSequenceClassification, BartForConditionalGeneration
import torch
from torch.utils.data import DataLoader

from textcat.ml.tokenizer.bert_tokenizer import GeneralTokenizer
from textcat.ml.tokenizer.adsorption_tokenizer import AdsorptionTokenizer
from textcat.ml.train_utils import run_inference_regression, run_inference_translation
from textcat.ml.dataset.regression_dataset import AdsorptionRegressionDataset
from textcat.ml.dataset.translation_dataset import AdsorptionTranslationDataset


class AdsorptionPredictor:
    def __init__(self, 
                 seq2seq_model: Union[str, torch.nn.Module]=None, 
                 seq2num_model:  Union[str, torch.nn.Module]=None, 
                 vocab_path: str = None, 
                 remove_tilde: bool = False, 
                 xval: bool = False, 
                 scaler = None, 
                 max_len: int = 256, 
                 verbose: int = 0):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if seq2seq_model:
            if isinstance(seq2seq_model, str):
                self.relaxer = BartForConditionalGeneration.from_pretrained(seq2seq_model, local_files_only=True)
            elif isinstance(seq2seq_model, torch.nn.Module):
                self.relaxer = seq2seq_model
            else:
                raise Exception()
            self.direct = False
            self.relaxer.to(self.device)
        else:
            self.relaxer = None
            self.direct = True

        if seq2num_model:
            if isinstance(seq2num_model, str):
                self.predictor = BertForSequenceClassification.from_pretrained(seq2num_model, num_labels=1)
            elif isinstance(seq2num_model, torch.nn.Module):
                self.predictor = seq2num_model
            else:
                raise Exception()
            self.predictor.to(self.device)
        else:
            raise Exception()
        
        self.scaler = torch.load(scaler) if scaler else None
        self.basic_tokenizer = AdsorptionTokenizer(remove_tilde, xval)
        self.tokenizer = GeneralTokenizer(vocab_path, self.basic_tokenizer)
        self.max_length = max_len
        self.verbose = verbose

    def tokenize(self, x: str):
        return self.tokenizer(x, 
                               None,
                               add_special_tokens=True,
                               padding='max_length',
                               max_length=self.max_length,
                               return_token_type_ids=False,
                               truncation=True)

    def infer(self, 
              x: Union[str, list[str]], 
              batch_size: int) -> Union[float, list[float]]:
        """
        Predict adsorption energy (in eV, ref. OC20)
        from the given SMILES.
        """
        if isinstance(x, str):
            input = self.tokenize(x)
            input_ids = torch.tensor(input["input_ids"], dtype=torch.long).unsqueeze(0).to(self.device)
            attention_mask = torch.tensor(input["attention_mask"], dtype=torch.long).unsqueeze(0).to(self.device)
            with torch.no_grad():
                eads_scaled = self.predictor(input_ids, attention_mask)['logits'].cpu().numpy()
                eads_rescaled = self.scaler.inverse_transform(eads_scaled)
                return eads_rescaled.item()
        else:
            y = []
            dataset = AdsorptionRegressionDataset(x, [], self.tokenizer, self.max_length)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            run_inference_regression(loader, self.device, self.predictor, y, self.scaler)
            return y

    def translate(self, x: Union[str, list[str]], 
                  batch_size: int) -> Union[str, list[str]]:
        if isinstance(x, str):
            input = self.tokenize(x)
            input_ids = torch.tensor(input["input_ids"], dtype=torch.long).unsqueeze(0).to(self.device)
            attention_mask = torch.tensor(input["attention_mask"], dtype=torch.long).unsqueeze(0).to(self.device)
            y = self.relaxer(input_ids, attention_mask)['logits'].cpu()  # Get model logits
            pred_ids = torch.argmax(y, dim=-1)  # Get predicted token ids
            decoded_preds = self.tokenizer.batch_decode(pred_ids, 
                                                skip_special_tokens=True, 
                                                clean_up_tokenization_spaces=True)
            relaxed_smiles = [y.replace(" ", "").strip() for y in decoded_preds]
            return relaxed_smiles[0]
        else:
            y = []
            dataset = AdsorptionTranslationDataset(x, [], self.tokenizer, self.max_length)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            run_inference_translation(loader, self.device, self.relaxer, self.tokenizer, y)
            return y 

    def __call__(self, 
                 x: Union[str, list[str]], 
                 batch_size: int = 128) -> dict[str, Union[str, float]]:
        """
        If direct strategy, directly infer Eads, else
        first predict relaxed SMILES and then infer Eads from it.
        """
        if self.direct:
           return {'is': x, 'rs': None, 'eads_eV': self.infer(x, batch_size)}
        else:
            rs = self.translate(x, batch_size)
            # Check validity
            if self.verbose != 0:
                if Chem.MolFromSmiles(rs, sanitize=False):
                    print("Relaxation successful (valid SMILES)!")
                else:
                    print("Relaxation lead to corrupt SMILES...")
            return {'is': x, 'rs': rs, 'eads_eV': self.infer(rs, batch_size)}
