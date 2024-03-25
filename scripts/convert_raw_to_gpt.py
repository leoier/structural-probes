import torch
from transformers import GPT2Tokenizer, GPT2Model
from argparse import ArgumentParser
import h5py
import numpy as np

argp = ArgumentParser()
argp.add_argument('input_path')
argp.add_argument('output_path')
argp.add_argument('gpt2_model', help='small, medium, large, or xl')
args = argp.parse_args()

# Map model argument to actual model sizes
model_size_map = {
    'small': 'gpt2',
    'medium': 'gpt2-medium',
    'large': 'gpt2-large',
    'xl': 'gpt2-xl'
}

model_name = model_size_map.get(args.gpt2_model)

if model_name is None:
    raise ValueError("GPT-2 model must be small, medium, large, or xl")

# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2Model.from_pretrained(model_name, output_hidden_states=True)

# Getting the number of layers and features from the model config
LAYER_COUNT = model.config.n_layer
FEATURE_COUNT = model.config.n_embd

model.eval()

with h5py.File(args.output_path, 'w') as fout:
    for index, line in enumerate(open(args.input_path)):
        if index % 100 == 0:
            print(f"Processing line {index}")
        line = line.strip()  # Remove trailing characters
        
        tokenized_text = tokenizer.tokenize(line)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])

        # tokens_tensor = tokenizer.encode(line, return_tensors='pt')

        with torch.no_grad():
            outputs = model(tokens_tensor)
            # In Transformers v4.x and above, outputs are returned as tuples
            hidden_states = outputs.hidden_states

        dset = fout.create_dataset(str(index), (LAYER_COUNT, len(tokenized_text), FEATURE_COUNT))
        dset[:, :, :] = np.vstack([np.array(x) for x in hidden_states[1:]])
