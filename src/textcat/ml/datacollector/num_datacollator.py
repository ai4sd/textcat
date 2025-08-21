"""
Data collator for dealing with xVal-encoded numbers.
"""

import torch

def define_masked_num_collator(tokenizer, mlm_probability):
    def masked_num_collator(batch):
        input_ids = torch.stack([sample["input_ids"] for sample in batch])
        input_num = torch.stack([sample["input_num"] for sample in batch])
        
        
        probability_matrix = torch.full(input_ids.shape, mlm_probability)
        mask = torch.bernoulli(probability_matrix).bool()
        
        labels = input_ids.clone()
        labels_num =input_num.clone()
        labels[~mask] = -100
        input_ids[mask] = tokenizer.mask_token_id
        input_num[mask] = 1
            
        return {'input_ids': input_ids, 
                'input_num':input_num, 
                'labels':labels, 
                'labels_num':labels_num}

    return masked_num_collator