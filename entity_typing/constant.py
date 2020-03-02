import os

from collections import defaultdict

import config_parser


config = config_parser.parser.parse_args()
GOAL = config.goal


def load_vocab_dict(vocab_file_name, vocab_max_size=None, start_vocab_count=None, common_vocab_file_name=None):
  with open(vocab_file_name) as f:
    text = [x.strip() for x in f.readlines()]
    if vocab_max_size:
      text = text[:vocab_max_size]
    if common_vocab_file_name:
        print('==> adding common training set types')
        print('==> before:', len(text))
        with open(common_vocab_file_name, 'r') as fc:
            common = [x.strip() for x in fc.readlines()]
        print('==> common:', len(common))
        text = list(set(text + common))
        print('==> after:', len(text))
    if start_vocab_count:
      file_content = dict(zip(text, range(0 + start_vocab_count, len(text) + start_vocab_count)))
    else:
      file_content = dict(zip(text, range(0, len(text))))
  return file_content


def load_definition_dict(path):
  with open(path, 'r') as f:
    definition = [[y.strip() for y in x.strip().split('<sep>')] for x in f.readlines()]
    definition = {k:v.strip().split() for k,v in definition}
  return definition


def get_definition_vocab(def_dict):
  counts = {}
  for _, v in def_dict.items():
    for word in v:
      if word not in counts:
        counts[word] = 0
      counts[word] += 1
  vocab = {'<unk>': 0, '<pad>': 1}
  idx = 2
  for k, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True):
    vocab[k] = idx
    idx += 1
  return vocab

# PATHs -- Modify these in your environment!
BASE_PATH = './'
FILE_ROOT = BASE_PATH + '../data/'
EXP_ROOT = BASE_PATH + '../models/'
OUT_ROOT = BASE_PATH + '../output/'

# Create output dir if it doesn't exist
if not os.path.exists(OUT_ROOT):
  os.mkdir(OUT_ROOT)

# CATEGORY VOCAB SIZE
ANSWER_NUM_DICT = {'conll_60k': 60000, 'unseen_60k': 60000}
ANSWER_NUM = ANSWER_NUM_DICT[GOAL]

# CATEGORY VOCAB
if GOAL == 'conll_60k':
  ANS2ID_DICT = load_vocab_dict(FILE_ROOT + "ontology/conll_categories.txt", vocab_max_size=60000)
  id2ans = {v: k for k, v in ANS2ID_DICT.items()}
  ID2ANS_DICT = id2ans
elif GOAL == 'unseen_60k':
  ANS2ID_DICT = load_vocab_dict(FILE_ROOT + "ontology/unseen_categories.txt", vocab_max_size=60000)
  id2ans = {v: k for k, v in ANS2ID_DICT.items()}
  ID2ANS_DICT = id2ans
else:
  raise NotImplementedError

# ELMo
ELMO_OPTIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
ELMO_WEIGHT_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

TYPE_BOS_IDX = ANSWER_NUM + 1
TYPE_EOS_IDX = ANSWER_NUM + 2
TYPE_PAD_IDX = ANSWER_NUM + 3

CHAR_DICT = defaultdict(int)
char_vocab = [u"<unk>"]
with open(FILE_ROOT + "/ontology/char_vocab.english.txt") as f:
  char_vocab.extend(c.strip() for c in f.readlines())
  CHAR_DICT.update({c: i for i, c in enumerate(char_vocab)})

# Printing constants
print('-- Constants ' + '-' * 67)
print('{:>20} : {:}'.format('GOAL', GOAL))
print('{:>20} : {:}'.format('BASE_PATH', BASE_PATH))
print('{:>20} : {:}'.format('FILE_ROOT', FILE_ROOT))
print('{:>20} : {:}'.format('EXP_PATH', EXP_ROOT))
print('{:>20} : {:}'.format('ANSWER_NUM', ANSWER_NUM))
print('{:>20} : {:}'.format('ANS2ID_DICT', len(ANS2ID_DICT)))
print('{:>20} : {:}'.format('ID2ANS_DICT', len(ID2ANS_DICT)))
print('{:>20} : {:}'.format('CHAR_DICT', len(CHAR_DICT)))