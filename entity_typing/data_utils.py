import glob
import json
import logging
import numpy as np
import string
import torch
from allennlp.commands.elmo import ElmoEmbedder
from random import shuffle
from nltk.corpus import stopwords

import constant


def to_torch(feed_dict, device):
    torch_feed_dict = {}
    ex_ids = None
    if 'ex_ids' in feed_dict:
        ex_ids = feed_dict.pop('ex_ids')
    for k, v in feed_dict.items():
        if 'embed' in k:
            torch_feed_dict[k] = torch.from_numpy(v).to(device).float()
            torch_feed_dict[k].requires_grad = False
        elif 'token_bio' == k:
            torch_feed_dict[k] = torch.from_numpy(v).to(device).float()
            torch_feed_dict[k].requires_grad = False
        elif 'y' == k or k == 'mention_start_ind' or k == 'mention_end_ind' or 'length' in k:
            torch_feed_dict[k] = torch.from_numpy(v).to(device)
            torch_feed_dict[k].requires_grad = False
        elif k in ['span_chars', 'candidates_emb_ids', 'candidates_ids', 'y_loc', 'token_ids', 'mention_ids', 'left_ids',
                   'right_ids', 'candidates_title_ids', 'candidates_lead_paragraph_ids',
                   'candidates_category_ids', 'bert_input_idx', 'bert_token_type_idx',
                   'bert_attention_mask']:
            torch_feed_dict[k] = torch.from_numpy(v).to(device)
            torch_feed_dict[k].requires_grad = False
        else:
            torch_feed_dict[k] = torch.from_numpy(v).to(device)
            torch_feed_dict[k].requires_grad = True
    return torch_feed_dict, ex_ids


def pad_slice(seq, seq_length, cut_left=False, pad_token="<none>"):
    if len(seq) >= seq_length:
        if not cut_left:
            return seq[:seq_length]
        else:
            output_seq = [x for x in seq if x != pad_token]
            if len(output_seq) >= seq_length:
                return output_seq[-seq_length:]
            else:
                return [pad_token] * (seq_length - len(output_seq)) + output_seq
    else:
        return seq + ([pad_token] * (seq_length - len(seq)))


def get_word_vec(word, vec_dict):
    if word in vec_dict:
        return vec_dict[word]
    return vec_dict['unk']


def get_w2v_idx(word, idx_dict):
    word = word.lower() # no need lower() for glove
    if word in idx_dict:
        return idx_dict[word]
    return idx_dict['<unk>']


def init_elmo():
    print('Preparing ELMo...')
    print("Loading options from {}...".format(constant.ELMO_OPTIONS_FILE)) 
    print("Loading weith from {}...".format(constant.ELMO_WEIGHT_FILE)) 
    return ElmoEmbedder(constant.ELMO_OPTIONS_FILE, constant.ELMO_WEIGHT_FILE, cuda_device=0)


def get_elmo_vec(sentence, elmo):
    """ sentence must be a list of words """
    emb = elmo.embed_sentence(sentence)
    #n_layer, _, _ = emb.shape
    #emb = emb.sum(0) / float(n_layer)
    return emb # (3, len, dim)


def get_elmo_vec_batch(sentences, elmo):
    """ sentence must be a list of words """
    emb = elmo.embed_batch(sentences)
    #averaged = []
    #for x in emb:
    #  n_layer, _, _ = x.shape
    #  averaged.append(x.sum(0) / float(n_layer))
    return emb  # (batch, 3, len, dim)


def get_bert_vec_batch(input_dict, bert):
    return bert(input_dict, None) # set data_type arg None, it's not used


def get_example(generator, batch_size, elmo, eval_data=False, answer_num=29932):
  # [cur_stream elements]
  # 0: example id, 1: left context, 2: right context, 3: mention word, 4: mention char,
  # 5: gold category
  use_elmo_batch = True  # Use elmo batch. Set this False if hitting GPU memory error.
  embed_dim = 1024
  cur_stream = [None] * batch_size
  no_more_data = False
  while True:
    bsz = batch_size
    seq_length = 200
    mention_length_limit = 100
    mention_char_limit = 1000
    for i in range(batch_size):
      try:
        cur_stream[i] = list(next(generator))
      except StopIteration:
        no_more_data = True
        bsz = i
        break
    max_seq_length = min(2*seq_length+mention_length_limit, max([len(elem[1]) + len(elem[2]) + len(elem[3]) for elem in cur_stream if elem]))
    token_seq_length = np.zeros([bsz], np.float32)
    token_bio = np.zeros([bsz, max_seq_length, 4], np.float32)
    max_mention_length = min(mention_length_limit, max([len(elem[3]) for elem in cur_stream if elem]))
    max_span_chars = min(mention_char_limit, max(max([len(elem[5]) for elem in cur_stream if elem]), 5))
    ex_ids = np.zeros([bsz], np.object)
    span_chars = np.zeros([bsz, max_span_chars], np.int64)
    targets = np.zeros([bsz, answer_num], np.float32)
    mention_span_length = np.zeros([bsz], np.float32)
    token_embed = np.zeros([bsz, 3, max_seq_length, embed_dim], np.float32)
    mention_embed = np.zeros([bsz, 3, max_mention_length, embed_dim], np.float32)
    # Batch to ELMo embeddings
    # Might get CUDA memory error if batch size is large
    if use_elmo_batch:
      token_seqs = []
      keys = []
      for i in range(bsz):
        left_seq = cur_stream[i][1]
        if len(left_seq) > seq_length:
          left_seq = left_seq[-seq_length:]
        mention_seq = cur_stream[i][3]
        right_seq = cur_stream[i][2]
        seq_ = left_seq + mention_seq + right_seq
        if len(seq_) > max_seq_length:
          seq_ = seq_[:max_seq_length]
        token_seqs.append(seq_)  # limit length to avoid GPU memory error
        keys.append(cur_stream[i][0])
      try:
        elmo_emb_batch = get_elmo_vec_batch(token_seqs, elmo)  # (batch, 3, len, dim)
      except:
        print('ERROR in get_elmo_vec_batch:', bsz, token_seqs, left_seq, mention_seq, right_seq)
        raise
    # Process each example
    for i in range(bsz):
      left_seq = cur_stream[i][1]
      if len(left_seq) > seq_length:
        left_seq = left_seq[-seq_length:]
      mention_seq = cur_stream[i][3]
      ex_ids[i] = cur_stream[i][0]
      right_seq = cur_stream[i][2]
      token_seq = left_seq + mention_seq + right_seq
      if len(token_seq) > max_seq_length:
        token_seq = token_seq[:max_seq_length]
      # sentence
      if use_elmo_batch:  # Faster
        elmo_emb = elmo_emb_batch[i]  # (3, len, dim)
      else:  # Slower, but ok with smaller GPUs
        elmo_emb = get_elmo_vec(token_seq, elmo)
      n_layers, seq_len, elmo_dim = elmo_emb.shape
      assert n_layers == 3, n_layers
      assert seq_len == len(token_seq), (seq_len, len(token_seq), token_seq, elmo_emb.shape)
      assert elmo_dim == embed_dim, (elmo_dim, embed_dim)
      if seq_len <= max_seq_length:
        token_embed[i, :n_layers, :seq_len, :] = elmo_emb
      else:
        token_embed[i, :n_layers, :, :] = elmo_emb[:, :max_seq_length, :]
      # mention span
      start_ind = len(left_seq)
      end_ind = len(left_seq) + len(mention_seq) - 1
      elmo_mention = elmo_emb[:, start_ind:end_ind+1, :]
      mention_len = end_ind - start_ind + 1
      assert mention_len == elmo_mention.shape[1] == len(mention_seq), (mention_len, elmo_mention.shape[1], len(mention_seq), mention_seq, elmo_mention.shape, token_seq, elmo_emb.shape) # (mention_len, elmo_mention.shape[0], len(mention_seq))
      if mention_len < max_mention_length:
        mention_embed[i, :n_layers, :mention_len, :] = elmo_mention
      else:
        mention_embed[i, :n_layers, :mention_len, :] = elmo_mention[:, :max_mention_length, :]
      for j, _ in enumerate(left_seq):
        token_bio[i, min(j, seq_length-1), 0] = 1.0  # token bio: 0(left) start(1) inside(2)  3(after)
      for j, _ in enumerate(right_seq):
        token_bio[i, min(j + len(mention_seq) + len(left_seq), seq_length-1), 3] = 1.0
      for j, _ in enumerate(mention_seq):
        if j == 0 and len(mention_seq) == 1:
          token_bio[i, min(j + len(left_seq), 49), 1] = 1.0
        else:
          token_bio[i, min(j + len(left_seq), 49), 2] = 1.0
      token_seq_length[i] = min(110, len(token_seq))
      # mention - char
      span_chars[i, :] = pad_slice(cur_stream[i][4], max_span_chars, pad_token=0)
      # targets (labels)
      for answer_ind in cur_stream[i][5]:
        targets[i, answer_ind] = 1.0
      mention_span_length[i] = min(len(mention_seq), 10)
    feed_dict = {"ex_ids": ex_ids,
                 "mention_embed": mention_embed,
                 "span_chars": span_chars,
                 "y": targets,
                 "mention_span_length": mention_span_length,
                 "token_bio": token_bio,
                 "token_embed": token_embed,
                 "token_seq_length": token_seq_length}
    if no_more_data:
      if eval_data and bsz > 0:
        yield feed_dict
      break
    yield feed_dict


class ETDataset(object):

  def __init__(self, filepattern, vocab, elmo=None, args=None):
    self.args = args
    self._all_shards = glob.glob(filepattern)
    shuffle(self._all_shards)
    self.char_vocab = vocab
    self.elmo = elmo
    self.word2id = constant.ANS2ID_DICT
    self.dropset = set(stopwords.words('english') + [c for c in string.punctuation])
    print('Found %d shards at %s' % (len(self._all_shards), filepattern))
    logging.info('Found %d shards at %s' % (len(self._all_shards), filepattern))

  def _load_npz(self, path):
    with open(path, 'rb') as f:
      data = np.load(f)
    return data

  def _drop_stopwords(self, s):
    dropped = [w for w in s if w.lower() not in self.dropset]
    if dropped:
      return dropped
    else:
      return ['<empty>'] # do this to avoid an empty sequence

  def _is_number(self, s):
    try:
      float(s)
      return True
    except:
      return False

  def _replace_numbers(self, s):
    return ['<number>' if self._is_number(w.lower()) else w  for w in s]

  def _load_shard(self, shard_name, eval_data):
    with open(shard_name) as f:
      print('==> Loarding: ' + shard_name)
      lines = [json.loads(line.strip()) for line in f.readlines()]
      ex_ids = [line["ex_id"] for line in lines]
      mention_char = [[self.char_vocab[x] for x in list(line["word"])] for line in lines]
      mention_word = [line["word"].split() if not self.args.annonym_mention else ['<mention>'] for line in lines]
      left_seq = [line['left_context'] if not self.args.drop_stopwords else
                  self._drop_stopwords(line['left_context'])  for line in lines]
      right_seq = [line['right_context'] if not self.args.drop_stopwords else
                   self._drop_stopwords(line['right_context']) for line in lines]

      if self.args.replace_numbers:
        left_seq = [self._replace_numbers(s) for s in left_seq]
        right_seq = [self._replace_numbers(s) for s in right_seq]

      y_categories = [line['y_category'] for line in lines]
      y_category_ids = []

      for iid, y_strs in enumerate(y_categories):
        y_category_ids.append([self.word2id[x] for x in y_strs if x in self.word2id])

      # 0: example id, 1: left context, 2: right context, 3: mention word, 4: mention char, 5: gold category
      return zip(ex_ids, left_seq, right_seq, mention_word, mention_char, y_category_ids)


  def _get_sentence(self, epoch, forever, eval_data):
    for i in range(0, epoch if not forever else 100000000000000):
      for shard in self._all_shards:
        ids = self._load_shard(shard, eval_data)
        #print('ids', list(ids))
        for current_ids in ids:
          yield current_ids

  def get_batch(self, batch_size=128, epoch=5, forever=False, eval_data=False):
    return get_example(
      self._get_sentence(epoch, forever=forever, eval_data=eval_data),
      batch_size,
      self.elmo,
      eval_data=eval_data,
      answer_num=constant.ANSWER_NUM_DICT[self.args.goal]
    )