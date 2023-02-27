import os
import json
import logging
import urllib.request
import torch

TEMPERATURE = 0.91
TOP_K = 0
TOP_P = 0.7
NO_SAMPLE = True

SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<speaker1>', '<speaker2>']}
MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

PERSONACHAT_URL = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"

logger = logging.getLogger(__file__)

def get_dataset(tokenizer, dataset_path = "", dataset_cache = ""):
    """ Get tokenized PERSONACHAT dataset from S3 or cache."""
    dataset_file = "personachat_self_original"
    dataset_path = dataset_path or PERSONACHAT_URL
    dataset_cache = dataset_cache or dataset_file + '_' + type(tokenizer).__name__  # To avoid using GPT cache for GPT-2 and vice-versa
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Download dataset from %s", dataset_path)
        try:
            urllib.request.urlretrieve(PERSONACHAT_URL, dataset_file)
            print('File downloaded successfully.')
        except urllib.error.URLError as e:
            print('Error downloading file:', e.reason)
            quit()

        with open(dataset_file, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())

        logger.info("Tokenize and encode the dataset")
        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        
        dataset = tokenize(dataset)
        torch.save(dataset, dataset_cache)
    return dataset

def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)