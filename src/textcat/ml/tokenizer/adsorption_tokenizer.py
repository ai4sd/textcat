"""
Module containing tokenizer for adsorption SMILES.
"""

import re

import numpy as np
from tqdm import tqdm 

class AdsorptionTokenizer:
    """
    Tokenizer class for SMILES adsorption text representation.
    """

    def __init__(self, 
                 remove_tilde: bool=False, 
                 xval: bool=False):
        """
        SMILES Tokenizer for adsorption structures.

        Parameters
        ----------
        remove_tilde(bool): If True, all "~" undefined bonds will be removed and considered as single.
        xval(bool): If True tokenize bond lengths found in the input SMILES string.
        """

        self.remove_tilde = remove_tilde
        self.xval = xval                
        self.pattern = r"""(\[[^\]]+]|Br?|Cl?|N|O|H|S|P|F|I|V|Ga|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9]|\{\d+\.\d+\})"""
        self.regex = re.compile(self.pattern)

    def tokenize(self, 
                 x: str) -> list[str]:
        """
        Tokenize HetSMILES.
        """

        if self.remove_tilde:
            x = x.replace("~", "")
            
        if self.xval:
            tokens = [token for token in self.regex.findall(x)]
            for i, tok in enumerate(tokens):
                if tok.startswith('{') and tok.endswith('}'):
                    tokens[i] = "[NUM]"
        else:
            tokens = [token for token in self.regex.findall(x)]  
        return tokens
    
    def extract_numbers(self, 
                        x: str) -> list[float]:
        """
        Extract numbers from HetSMILES string.
        """
        numbers= []
        for tok in [token for token in self.regex.findall(x)]:
            if tok.startswith('{') and tok.endswith('}'):
                numbers.append(float(tok[1:-1]))
    
        return numbers
    
    def generate_vocabulary(self, 
                            xl: list[str], 
                            output_name: str=None) -> dict[int, str]:
        """Generate vocabulary given a list of strings."""

        vocab = set()
        for x in tqdm(xl):
            tokens = self.tokenize(x)
            for token in tokens:
                vocab.add(token)

        # Special tokens
        vocab.add("[UNK]")  # Unknown token 
        vocab.add("[SEP]")  # Separator token
        vocab.add("[PAD]")  # Padding token
        vocab.add("[CLS]")  # Special initial token
        vocab.add("[MASK]") # Masking token

        vocab_dict = {}
        for i, token in enumerate(sorted(list(vocab))):
            vocab_dict[i] = token

        if output_name:  # store vocabulary as text file
            with open(output_name, "w", encoding="utf-8") as writer:
                for token in vocab_dict.values():
                    writer.write(token+"\n")

        return vocab_dict

    def evaluate_max_len(self, xs: list[str]) -> dict:
        """
        Tokenize all the strings in the input dataset list
        and return statistics on sequence length. 
        Useful for defining max embedding size 
        when training sequence models.
        """
        lengths = []
        for x in tqdm(xs):
            lengths.append(len(self.tokenize(x)))
        out = {}
        out['max'] = max(lengths)
        out['min'] = min(lengths)
        out['mean'] = sum(lengths) / len(lengths)
        out['median'] = np.median(lengths)
        percentiles = np.percentile(lengths, [25, 50, 75, 90, 95, 99])
        out['q25'] = percentiles[0]
        out['q50'] = percentiles[1]
        out['q75'] = percentiles[2]
        out['q90'] = percentiles[3]
        out['q95'] = percentiles[4]
        out['q99'] = percentiles[5]
        out['lengths'] = lengths
        return out


