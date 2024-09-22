from datasets import Features, Sequence, Value
import numpy as np
import torch
from torch.utils.data import DataLoader

def tokenize_dataset(dataset, tokenizer, text_column_name, sequence_length, dataset_processing_num_proc_per_process):
    """tokenize a dataset and split it into chunks of size sequence_length + 1
    Args:
        dataset (_type_): HF dataset
        column_name (_type_): the name of the column to tokenize
    Returns:
        _type_: a dictionary containing the tokenized text in chunks of size sequence_length + 1
    """
    def _tokenizer_group_text(texts, sequence_length): # texts is a list of dataset rows.
        tokenized_text_batch = tokenizer.batch_encode_plus(texts,return_attention_mask=False, return_token_type_ids=False, return_tensors='np' ) # list of np.arrays {'input_ids': [[],[]...]}
        concatenated_tokens = {'input_ids': np.concatenate(tokenized_text_batch['input_ids'])} # concatenate all the np.arrays(all the text)
        total_length = len(concatenated_tokens['input_ids']) # get the total length of the concatenated tokens
        if total_length >= sequence_length + 1:
            total_length = ((total_length - 1) // sequence_length) * sequence_length + 1
        result = {
            'input_ids': [
                concatenated_tokens['input_ids'][i : i + sequence_length + 1] for i in range(0, total_length - sequence_length, sequence_length) 
            ]
        }
        return result # {'input_ids': [[],[]...]} each chunk of size sequence_length + 1
    
    tokenized_dataset = dataset.map(
        _tokenizer_group_text,
        input_columns=text_column_name,
        remove_columns=dataset.column_names,
        features=Features({"input_ids": Sequence(feature=Value(dtype="int64"), length=sequence_length + 1)}),
        batched=True,
        num_proc=dataset_processing_num_proc_per_process,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {sequence_length+1}",
        fn_kwargs={"sequence_length": sequence_length},
    )
    
    return tokenized_dataset


def get_dataloader(tokenized_dataset, batch_size, shuffle, sampler = None, pin_memory=True, num_workers=0):
    """Create a dataloader from a tokenized dataset
    Args:
        tokenized_dataset (_type_): a tokenized dataset
        batch_size (_type_): the batch size
        shuffle (_type_): whether to shuffle the dataset
        collate_fn (_type_): the collate function
    Returns:
        _type_: a dataloader
    """
    def collate_fn(batch):
        input_ids = [item['input_ids'][:-1] for item in batch]
        label_ids = [item['input_ids'][1:] for item in batch]
        attention_mask = [[1] * len(input_id) for input_id in input_ids] # just all 1 for now
        label_mask = [[1] * len(label_id) for label_id in label_ids]
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long), # merge all the input_ids into a single tensor
            'label_ids': torch.tensor(label_ids, dtype=torch.long), 
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'label_mask': torch.tensor(label_mask, dtype=torch.long),
        }
    return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, sampler= sampler, num_workers=num_workers, pin_memory=pin_memory)