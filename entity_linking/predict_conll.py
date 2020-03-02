import argparse
import json
import numpy as np
import pickle as pkl
import random
from tqdm import tqdm

random.seed(123)

parser = argparse.ArgumentParser()
parser.add_argument("-goal", help="category vocab.", default="unseen_60k", choices=["conll_60k"])
parser.add_argument("-popular_prior_path", help="Dev data", default="data/resources/popular_prior_wiki.tsv")
parser.add_argument("-root_path", help="Root dir.", default="../")
parser.add_argument("-data_path", help="Path to a data file.",
                    default="data/entity_linking_data/validation/dev_et4el_conll_60k_with_candidates.json")
parser.add_argument("-type_probs_path", help="ET model output.",
                    default="output/conll_60k_eval_dev.pkl")
parser.add_argument("-category_set_path", help="Category set.",
                    default="data/ontology/conll_categories.txt")


def is_num(s):
  try:
    float(s)
    return True
  except:
    return False


def load_json(path):
  with open(path, 'r') as f:
    lines = f.readlines()
    return [json.loads(line.strip()) for line in lines]


def save_json(path, data):
  with open(path, 'w') as f:
    for d in data:
      json.dump(d, f)
      f.write('\n')


def load_vocab_dict(vocab_file_name, vocab_max_size=None, start_vocab_count=None):
  with open(vocab_file_name) as f:
    text = [x.strip() for x in f.readlines()]
    if vocab_max_size:
      text = text[:vocab_max_size]
    if start_vocab_count:
      file_content = dict(zip(text, range(0 + start_vocab_count, len(text) + start_vocab_count)))
    else:
      file_content = dict(zip(text, range(0, len(text))))
  return file_content


def upper_first(s):
  return s[0].upper() + s[1:]


def get_popular_prior(path):
  with open(path, 'r') as f:
    popular_prior = [l.strip().split('\t') for l in tqdm(
      f.readlines(), desc='{:>30} : '.format('Loading Popular Prior'),
      bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}')]
    popular_prior = {x[0]: (int(x[1]), x[2]) for x in popular_prior if len(x) == 3}
    return popular_prior


def get_type_probs(path):
  with open(path, 'rb') as f:
    return pkl.load(f)['pred_dist']


def get_prediction(data, output_probs, popular_prior, category_set, ans2id_dict):
  assert len(data) == len(output_probs), (len(data), len(output_probs))
  correct = 0.
  prior_correct = 0.
  gold_pred = []
  i = 0
  for ex in tqdm(data,
    desc='{:>30} : '.format('Processing Examples'),
    bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}'):
    ex_id = ex['ex_id']
    row_id = int(ex_id.split('_')[0])  # Assuming row_id is {0, ..., N}
    if row_id < len(output_probs):
      entity_type_probs = output_probs[row_id]
      gold = int(ex['wikiId'])
      # check popular prior...
      m = ex['word']
      popular_prior_pred = None
      if m in popular_prior:
        popular_prior_pred = popular_prior[m][0]
        if gold == popular_prior_pred:
          prior_correct += 1.
      pred = None
      # Rank candidates
      if ex['candidates']:
        scores = []
        candidates = {d['wikiId']: d for d in ex['candidates']}
        for candidate in candidates.values():
          # Compute scores for each candidate
          if candidate['category'] and candidate['category'][0]:  # check if 'category' is empty ...
            candidate_categories = set([c.lower() for c in candidate['category'][0] if c.lower() in category_set])
            candidate_category_ids = sorted([ans2id_dict[c] for c in candidate_categories])
            score = 0.
            if candidate_category_ids:
              # Dot product between ET output vector and binary category vector (summing non zero elements)
              score = np.sum([s if s >= 0. else 0. for s in entity_type_probs[candidate_category_ids]])
            elif popular_prior_pred is not None and int(popular_prior_pred) == int(candidate['wikiId']):
              # Use popular prior if categories are not available
              score = 1e+06
            else:
              score = 0.
            scores.append((candidate['wikiId'], score))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        score_nums = [tup[1] for tup in scores]
        if (len(score_nums) > 1) and (score_nums[0] > 0)\
          and (score_nums[1] > 0) and (score_nums[0] == score_nums[1]):
          # Tie: use popular prior
          highest_score = max(score_nums)
          if popular_prior_pred:
            pred = popular_prior_pred
          else:
            # Pick one randomly if popular prior is not available
            pred, _ = random.sample([s for s in scores if s[1] == highest_score], 1)[0]
          if gold == pred:
            correct += 1.
        elif sum(score_nums):
          # Pick a candidate with the highest score
          pred, _ = scores[0]
          if gold == pred:
            correct += 1.
        else:
          # Other cases: use popular prior
          if popular_prior_pred:
            pred = popular_prior_pred
          elif scores:
            # Pick one randomly if popular prior is not available
            pred, _ = random.sample(scores, 1)[0]
          else:
            # no valid candidates and no popular prior  --  no way to predict...
            pred = 0
          if gold == pred:
            correct += 1.
      else:
        if popular_prior_pred:
          pred = popular_prior_pred
          if gold == pred:
            correct += 1.
        else:
          # no candidates and no popular prior  --  no way to predict...
          pred = 0
      assert pred is not None, pred
      gold_pred.append({'ex_id': ex_id, 'gold': gold, 'pred': pred})
    else:
      print('ERROR: Invalid example index.')
      raise
    # if i > 0 and i % 1000 == 0:
    #   print('Processed {:>6} examples...'.format(i))
    i += 1
  assert len(data) == len(gold_pred), (len(data), len(gold_pred))
  return gold_pred, prior_correct


if __name__ == '__main__':
    config = parser.parse_args()
    entity_linking_data_path = config.root_path + config.data_path
    type_probs_path = config.root_path + config.type_probs_path
    category_set_path = config.root_path + config.category_set_path
    print('{:>30} : {:}'.format('Category Vocab', config.goal))
    print('{:>30} : {:}'.format('Entity Linking Data', entity_linking_data_path))
    print('{:>30} : {:}'.format('Entity Typing Output', type_probs_path))
    print('{:>30} : {:}'.format('Category Set File', category_set_path))
    entity_linking_data = load_json(entity_linking_data_path)
    popular_prior = get_popular_prior(config.root_path + config.popular_prior_path)
    type_probs = get_type_probs(type_probs_path)
    ans2id_dict = load_vocab_dict(category_set_path, vocab_max_size=60000)
    category_set = set(ans2id_dict.keys())
    print('{:>30} : {:}'.format('Data Size', len(entity_linking_data)))
    print('{:>30} : {:}'.format('Output Size', len(type_probs)))
    gold_pred, prior_correct = get_prediction(entity_linking_data, type_probs, popular_prior, category_set, ans2id_dict)
    save_to = entity_linking_data_path + '.pred'
    save_json(save_to, gold_pred)
    # Print accuracy
    et4el_accuracy = sum(gp['gold'] == gp['pred'] for gp in gold_pred) / float(len(gold_pred))
    popular_prior_accuracy = prior_correct / float(len(gold_pred))
    print('{:>30} : {:}'.format('Prediction File', save_to))
    print('{:>30} : {:.4f}'.format('Popular Prior Accuracy', popular_prior_accuracy))
    print('{:>30} : {:.4f}'.format('ET4EL Accuracy', et4el_accuracy))
