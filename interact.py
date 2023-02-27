from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, GPT2Tokenizer, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
import argparse, random, torch
import torch.nn.functional as F
from itertools import chain

from utils import TEMPERATURE, TOP_K, TOP_P, NO_SAMPLE, SPECIAL_TOKENS, ATTR_TO_SPECIAL_TOKEN, get_dataset, add_special_tokens_

modelpath = "" # use last model from runs folder!

def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits

def build_input_from_segments(persona, history, reply, tokenizer, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    sequence = [[bos] + list(chain(*persona))] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = [-100] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]
    return instance

def sample_sequence(personality, history, tokenizer, model):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    current_output = []

    for i in range(100):
        instance = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device="cuda").unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device="cuda").unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)
        #if isinstance(logits, tuple):  # for gpt2 and maybe others
        logits = logits[0]
        logits = logits[0, -1, :] / TEMPERATURE
        logits = top_filtering(logits, top_k=TOP_K, top_p=TOP_P)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if NO_SAMPLE else torch.multinomial(probs, 1)
        if i < 0 and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:                    
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output

# Define the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modelpath', help='Path to the model file', required=True)
args = parser.parse_args()

# Fill the modelpath variable
modelpath = args.modelpath

# Check if the modelpath is empty
if not modelpath:
    print('Error: --modelpath cannot be empty')
    quit()

print("use model: ", modelpath)

tokenizer = GPT2Tokenizer.from_pretrained(modelpath)
model = GPT2LMHeadModel.from_pretrained(modelpath)

model.to("cuda")

print("model loaded")

add_special_tokens_(model, tokenizer)
print("tokens added")

dataset = get_dataset(tokenizer)
personalities = [dialog["personality"] for dataset in dataset.values() for dialog in dataset]
personality = random.choice(personalities)

print("Selected personality: ", tokenizer.decode(chain(*personality)))

history = []
while True:
    raw_text = input(">>> ")
    while not raw_text:
        print('Prompt should not be empty!')
        raw_text = input(">>> ")

    history.append(tokenizer.encode(raw_text))
    with torch.no_grad():
        out_ids = sample_sequence(personality, history, tokenizer, model)

    history.append(out_ids)
    history = history[-(20 + 1):]
    out_text = tokenizer.decode(out_ids, skip_special_tokens=True)

    print(out_text)