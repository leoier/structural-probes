import torch
from transformers import GPT2Tokenizer, GPT2Model, RobertaTokenizer, RobertaModel
from argparse import ArgumentParser
import h5py
import numpy as np
from tqdm import tqdm

argp = ArgumentParser()
argp.add_argument('input_path')
argp.add_argument('output_path')
argp.add_argument('model_name')
args = argp.parse_args()

# Load pre-trained model tokenizer (vocabulary)
if args.model_name.startswith('gpt2-'):
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    model = GPT2Model.from_pretrained(args.model_name, output_hidden_states=True)
    LAYER_COUNT = model.config.n_layer
    FEATURE_COUNT = model.config.n_embd
elif args.model_name.startswith('roberta-'):
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name, add_special_tokens=True)
    model = RobertaModel.from_pretrained(args.model_name, output_hidden_states=True)
    LAYER_COUNT = model.config.num_hidden_layers
    FEATURE_COUNT = model.config.hidden_size
else:
    raise ValueError(f"Unknown model name: {args.model_name}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Getting the number of layers and features from the model config

model.eval()

total_lines = sum(1 for _ in open(args.input_path))

with h5py.File(args.output_path, 'w') as fout, open(args.input_path) as fin:
    for index, line in tqdm(enumerate(fin), total=total_lines, desc="Processing lines"):
        line = line.strip()  # Remove trailing characters
        tokenized_text = tokenizer.tokenize(line)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens]).to(device)

        with torch.no_grad():
            outputs = model(tokens_tensor)
            # In Transformers v4.x and above, outputs are returned as tuples
            hidden_states = outputs.hidden_states

        dset = fout.create_dataset(str(index), (LAYER_COUNT+1, len(tokenized_text), FEATURE_COUNT))
        dset[:, :, :] = np.vstack([np.array(x.cpu()) for x in hidden_states])
