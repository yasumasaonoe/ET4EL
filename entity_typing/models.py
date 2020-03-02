import torch
import torch.nn as nn

from model_utils import sort_batch_by_length, SelfAttentiveSum, SimpleDecoder, CNN, ELMoWeightedSum
from model_utils import BCEWithLogitsLossCustom
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ModelBase(nn.Module):
  def __init__(self, args):
    super(ModelBase, self).__init__()

    self.custom_loss = args.custom_loss
    self.loss_neg_weight = args.loss_neg_weight

    if not self.custom_loss:
      self.loss_func = nn.BCEWithLogitsLoss()
    else:
      print('==> Model: usinf custom loss with neg weight =', self.loss_neg_weight)
      self.loss_func = BCEWithLogitsLossCustom()

  def sorted_rnn(self, sequences, sequence_lengths, rnn):
    sorted_inputs, sorted_sequence_lengths, restoration_indices = sort_batch_by_length(sequences, sequence_lengths)
    packed_sequence_input = pack_padded_sequence(sorted_inputs,
                                                 sorted_sequence_lengths.data.long().tolist(),
                                                 batch_first=True)
    packed_sequence_output, _ = rnn(packed_sequence_input, None)
    unpacked_sequence_tensor, _ = pad_packed_sequence(packed_sequence_output, batch_first=True)
    return unpacked_sequence_tensor.index_select(0, restoration_indices)

  def rnn(self, sequences, lstm):
    outputs, _ = lstm(sequences)
    return outputs.contiguous()

  def define_loss(self, logits, targets):
    loss = self.loss_func(logits, targets)
    return loss

  def define_custom_loss(self, logits, targets):
    neg_weight = torch.ones_like(logits) * self.loss_neg_weight # negative weight tensor
    loss = self.loss_func(logits, targets, neg_weight=neg_weight)
    return loss

  def forward(self, feed_dict):
    pass


class ETModel(ModelBase):
  def __init__(self, args, answer_num):
    super(ETModel, self).__init__(args)
    self.multi_gpu = args.multi_gpu
    self.output_dim = args.rnn_dim * 2
    self.mention_dropout = nn.Dropout(args.mention_dropout)
    self.input_dropout = nn.Dropout(args.input_dropout)
    self.dim_hidden = args.dim_hidden
    self.embed_dim = 1024
    self.mention_dim = 1024
    self.headword_dim = 1024
    self.enhanced_mention = args.enhanced_mention
    self.mention_lstm = args.mention_lstm
    self.annonym_mention = args.annonym_mention
    if self.annonym_mention:
      self.enhanced_mention = False
      self.mention_lstm = False
    if args.enhanced_mention:
      self.head_attentive_sum = SelfAttentiveSum(self.mention_dim, 1)
      self.cnn = CNN()
      self.mention_dim += 50
    if not self.annonym_mention:
      self.output_dim += self.mention_dim
    # Defining LSTM here.  
    self.attentive_sum = SelfAttentiveSum(args.rnn_dim * 2, 100)
    self.lstm = nn.LSTM(self.embed_dim + 50, args.rnn_dim, bidirectional=True, batch_first=True)
    self.token_mask = nn.Linear(4, 50)
    if self.mention_lstm:
      self.lstm_mention = nn.LSTM(self.embed_dim, self.embed_dim // 2, bidirectional=True, batch_first=True)
      self.mention_attentive_sum = SelfAttentiveSum(self.embed_dim, 1)
    self.sigmoid_fn = nn.Sigmoid()
    self.goal = args.goal
    self.decoder = SimpleDecoder(self.output_dim, answer_num)
    self.weighted_sum = ELMoWeightedSum()

  def forward(self, feed_dict):
    token_embed = self.weighted_sum(feed_dict['token_embed'])
    token_mask_embed = self.token_mask(feed_dict['token_bio'].view(-1, 4))
    token_mask_embed = token_mask_embed.view(token_embed.size()[0], -1, 50)  # location embedding
    token_embed = torch.cat((token_embed, token_mask_embed), 2)
    token_embed = self.input_dropout(token_embed)
    context_rep = self.sorted_rnn(token_embed, feed_dict['token_seq_length'], self.lstm)
    context_rep, _ = self.attentive_sum(context_rep)
    # Mention Representation
    if not self.annonym_mention:
      mention_embed = self.weighted_sum(feed_dict['mention_embed'])
      if self.enhanced_mention:
        if self.mention_lstm:
          mention_hid = self.sorted_rnn(mention_embed, feed_dict['mention_span_length'], self.lstm_mention)
          mention_embed, attn_score = self.mention_attentive_sum(mention_hid)
        else:
          mention_embed, attn_score = self.head_attentive_sum(mention_embed)
        span_cnn_embed = self.cnn(feed_dict['span_chars'])
        mention_embed = torch.cat((span_cnn_embed, mention_embed), 1)
      else:
        mention_embed = torch.sum(mention_embed, dim=1)
      mention_embed = self.mention_dropout(mention_embed)
      output = torch.cat((context_rep, mention_embed), 1)
      attn_score = None
    else:
        output = context_rep
        attn_score = None
    logits = self.decoder(output)
    if self.multi_gpu:
        return logits
    else:
        loss = self.define_loss(logits, feed_dict['y']) if not self.custom_loss else self.define_custom_loss(logits, feed_dict['y'])
        return loss, logits, attn_score