""" Demos a trained structural probe by making parsing predictions on stdin """

from argparse import ArgumentParser
import os
from datetime import datetime
import shutil
import yaml
from tqdm import tqdm
import torch
import numpy as np
import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
mpl.rcParams['agg.path.chunksize'] = 10000

import data
import model
import probe
import regimen
import reporter
import task
import loss
import run_experiment

from transformers import GPT2TokenizerFast, GPT2Model


def print_tikz(args, prediction_edges, words):
  '''
  Turns edge sets on word (nodes) into tikz dependency charts.

  Args:
    prediction_edges: A set of index pairs representing undirected dependency edges
    words: A list of strings representing the sentence
  '''
  with open(os.path.join(args['reporting']['root'], 'demo.tikz'), 'a') as fout:
    string = """\\begin{dependency}[hide label, edge unit distance=.5ex]
  \\begin{deptext}[column sep=0.05cm]
  """
    string += "\\& ".join([x.replace('$', '\$').replace('&', '+') for x in words]) + " \\\\" + '\n'
    string += "\\end{deptext}" + '\n'
    for i_index, j_index in prediction_edges:
      string += '\\depedge[edge style={{red!60!}}, edge below]{{{}}}{{{}}}{{{}}}\n'.format(i_index+1,j_index+1, '.')
    string += '\\end{dependency}\n'
    fout.write('\n\n')
    fout.write(string)

def print_distance_image(args, words, prediction, sent_index):
  """Writes an distance matrix image to disk.

  Args:
    args: the yaml config dictionary
    words: A list of strings representing the sentence
    prediction: A numpy matrix of shape (len(words), len(words))
    sent_index: An index for identifying this sentence, used
        in the image filename.
  """
  plt.clf()
  ax = sns.heatmap(prediction)
  ax.set_title('Predicted Parse Distance (squared)')
  ax.set_xticks(np.arange(len(words)))
  ax.set_yticks(np.arange(len(words)))
  ax.set_xticklabels(words, rotation=90, fontsize=6, ha='center')
  ax.set_yticklabels(words, rotation=0, fontsize=6, va='center')
  plt.tight_layout()
  plt.savefig(os.path.join(args['reporting']['root'], 'demo-dist-pred'+str(sent_index)), dpi=300)

def print_depth_image(args, words, prediction, sent_index):
  """Writes an depth visualization image to disk.

  Args:
    args: the yaml config dictionary
    words: A list of strings representing the sentence
    prediction: A numpy matrix of shape (len(words),)
    sent_index: An index for identifying this sentence, used
        in the image filename.
  """
  plt.clf()
  fontsize = 6
  cumdist = 0
  for index, (word, pred) in enumerate(zip(words, prediction)):
    plt.text(cumdist*3, pred*2, word, fontsize=fontsize, color='red', ha='center')
    cumdist = cumdist + (np.square(len(word)) + 1)

  plt.ylim(0,20)
  plt.xlim(0,cumdist*3.5)
  plt.title('Predicted Parse Depth (squared)', fontsize=10)
  plt.ylabel('Tree Depth', fontsize=10)
  plt.xlabel('Linear Absolute Position',fontsize=10)
  plt.tight_layout()
  plt.xticks(fontsize=5)
  plt.yticks(fontsize=5)
  plt.savefig(os.path.join(args['reporting']['root'], 'demo-depth-pred'+str(sent_index)), dpi=300)

def report_on_stdin(args):
  """Runs a trained structural probe on sentences piped to stdin.

  Sentences should be space-tokenized.
  A single distance image and depth image will be printed for each line of stdin.

  Args:
    args: the yaml config dictionary
  """

  # Define the GPT-2 model and tokenizer
  tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
  model = GPT2Model.from_pretrained('gpt2', output_hidden_states=True)
  LAYER_COUNT = 12
  FEATURE_COUNT = 768
  model.to(args['device'])
  model.eval()

  # Define the distance probe
  distance_probe = probe.TwoWordPSDProbe(args)
  distance_probe.load_state_dict(torch.load(args['probe']['distance_params_path'], map_location=args['device']))

  # Define the depth probe
  depth_probe = probe.OneWordPSDProbe(args)
  depth_probe.load_state_dict(torch.load(args['probe']['depth_params_path'], map_location=args['device']))

  for index, line in tqdm(enumerate(sys.stdin), desc='[demoing]'):
    # Tokenize the sentence and create tensor inputs to GPT-2
    untokenized_sent = line.strip().split()
    enc = tokenizer(' '.join(untokenized_sent), return_tensors='pt', return_offsets_mapping=True)
    untok_tok_mapping = data.GPTDataset.convert_offsets_to_mapping(enc['offset_mapping'][0], untokenized_sent)
    tokenized_sent = tokenizer.tokenize(' '.join(untokenized_sent))
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_sent)
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    tokens_tensor = tokens_tensor.to(args['device'])

    with torch.no_grad():
      # Run sentence tensor through GPT-2 after averaging subwords for each token
      outputs = model(tokens_tensor)
      hidden_states = outputs.hidden_states
      single_layer_features = hidden_states[args['model']['model_layer']].detach().clone()
      representation = torch.stack([torch.mean(single_layer_features[0,untok_tok_mapping[i][0]:untok_tok_mapping[i][-1]+1,:], dim=0) for i in range(len(untokenized_sent))], dim=0)
      representation = representation.view(1, *representation.size())

      # Run token vectors through the trained probes
      distance_predictions = distance_probe(representation.to(args['device'])).detach().cpu()[0][:len(untokenized_sent),:len(untokenized_sent)].numpy()
      depth_predictions = depth_probe(representation).detach().cpu()[0][:len(untokenized_sent)].numpy()

      # Print results visualizations
      print_distance_image(args, untokenized_sent, distance_predictions, index)
      print_depth_image(args, untokenized_sent, depth_predictions, index)

      predicted_edges = reporter.prims_matrix_to_edges(distance_predictions, untokenized_sent, untokenized_sent)
      print_tikz(args, predicted_edges, untokenized_sent)


if __name__ == '__main__':
  argp = ArgumentParser()
  argp.add_argument('experiment_config')
  argp.add_argument('--results-dir', default='',
      help='Set to reuse an old results dir; '
      'if left empty, new directory is created')
  argp.add_argument('--seed', default=0, type=int,
      help='sets all random seeds for (within-machine) reproducibility')
  cli_args = argp.parse_args()
  if cli_args.seed:
    np.random.seed(cli_args.seed)
    torch.manual_seed(cli_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
  yaml_args= yaml.load(open(cli_args.experiment_config))
  run_experiment.setup_new_experiment_dir(cli_args, yaml_args, cli_args.results_dir)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  yaml_args['device'] = device
  report_on_stdin(yaml_args)
