import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

import constant
import eval_metric


sigmoid_fn = nn.Sigmoid()


def get_eval_string(true_prediction):
  """
  Given a list of (gold, prediction)s, generate output string.
  """
  count, pred_count, avg_pred_count, p, r, f1 = eval_metric.micro(true_prediction)
  _, _, _, ma_p, ma_r, ma_f1 = eval_metric.macro(true_prediction)
  output_str = "Eval: {0} {1} {2:.3f} P:{3:.3f} R:{4:.3f} F1:{5:.3f} Ma_P:{6:.3f} Ma_R:{7:.3f} Ma_F1:{8:.3f}".format(
    count, pred_count, avg_pred_count, p, r, f1, ma_p, ma_r, ma_f1)
  accuracy = sum([set(y) == set(yp) for y, yp in true_prediction]) * 1.0 / len(true_prediction)
  output_str += '\t Dev accuracy: {0:.1f}%'.format(accuracy * 100)
  return output_str


def get_output_index(outputs, threshold=0.5):
  """
  Given outputs from the decoder, generate prediction index.
  :param outputs:
  :return:
  """
  pred_idx = []
  outputs = sigmoid_fn(outputs).data.cpu().clone()
  for single_dist in outputs:
    single_dist = single_dist.numpy()
    arg_max_ind = np.argmax(single_dist)
    pred_id = [arg_max_ind]
    pred_id.extend(
      [i for i in range(len(single_dist)) if single_dist[i] > threshold and i != arg_max_ind])
    pred_idx.append(pred_id)
  return pred_idx


def get_gold_pred_str(pred_idx, gold, goal, cls_logits=None, y_cls=None, y_noisy_idx=None):
  """
  Given predicted ids and gold ids, generate a list of (gold, pred) pairs of length batch_size.
  """
  id2word_dict = constant.ID2ANS_DICT
  gold_strs = []
  for gold_i in gold:
    gold_strs.append([id2word_dict[i] for i in range(len(gold_i)) if gold_i[i] == 1])
  pred_strs = []
  for pred_idx1 in pred_idx:
    pred_strs.append([(id2word_dict[ind]) for ind in pred_idx1])
  if cls_logits is not None and y_cls is not None and y_noisy_idx is not None:
    cls_pred = []
    for cls in cls_logits:
      cls_pred.append(1 if cls >= 0. else 0)
    y_cls_ = []
    for yc in y_cls:
      y_cls_.append(int(yc))
    y_noisy_idx_ = []
    for yni in y_noisy_idx:
      y_noisy_idx_.append([(id2word_dict[int(ind)]) for ind in yni if int(ind) != constant.TYPE_PAD_IDX])
    return list(zip(gold_strs, pred_strs, cls_pred, y_cls_, y_noisy_idx_)) 
  else:
    return list(zip(gold_strs, pred_strs))


# Borrowed from AllenNLP
def sort_batch_by_length(tensor: torch.autograd.Variable, sequence_lengths: torch.autograd.Variable):
  """
  @ from allennlp
  Sort a batch first tensor by some specified lengths.

  Parameters
  ----------
  tensor : Variable(torch.FloatTensor), required.
      A batch first Pytorch tensor.
  sequence_lengths : Variable(torch.LongTensor), required.
      A tensor representing the lengths of some dimension of the tensor which
      we want to sort by.

  Returns
  -------
  sorted_tensor : Variable(torch.FloatTensor)
      The original tensor sorted along the batch dimension with respect to sequence_lengths.
  sorted_sequence_lengths : Variable(torch.LongTensor)
      The original sequence_lengths sorted by decreasing size.
  restoration_indices : Variable(torch.LongTensor)
      Indices into the sorted_tensor such that
      ``sorted_tensor.index_select(0, restoration_indices) == original_tensor``
  """

  if not isinstance(tensor, Variable) or not isinstance(sequence_lengths, Variable):
    raise ValueError("Both the tensor and sequence lengths must be torch.autograd.Variables.")

  sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
  sorted_tensor = tensor.index_select(0, permutation_index)
  # This is ugly, but required - we are creating a new variable at runtime, so we
  # must ensure it has the correct CUDA vs non-CUDA type. We do this by cloning and
  # refilling one of the inputs to the function.
  index_range = sequence_lengths.data.clone().copy_(torch.arange(0, len(sequence_lengths)))
  # This is the equivalent of zipping with index, sorting by the original
  # sequence lengths and returning the now sorted indices.
  index_range = Variable(index_range.long())
  _, reverse_mapping = permutation_index.sort(0, descending=False)
  restoration_indices = index_range.index_select(0, reverse_mapping)
  return sorted_tensor, sorted_sequence_lengths, restoration_indices


class MultiSimpleDecoder(nn.Module):
  """
    Simple decoder in multi-task setting.
  """

  def __init__(self, output_dim):
    super(MultiSimpleDecoder, self).__init__()
    self.linear = nn.Linear(output_dim, constant.ANSWER_NUM_DICT['open'],
                            bias=False).cuda()  # (out_features x in_features)  #### bias

  def forward(self, inputs, output_type):
    if output_type == "open":
      return self.linear(inputs)
    elif output_type == 'wiki':
      return F.linear(inputs, self.linear.weight[:constant.ANSWER_NUM_DICT['wiki'], :], self.linear.bias[:constant.ANSWER_NUM_DICT['wiki']])
      #return F.linear(inputs, self.linear.weight[:constant.ANSWER_NUM_DICT['wiki'], :])
    elif output_type == 'kb':
      return F.linear(inputs, self.linear.weight[:constant.ANSWER_NUM_DICT['kb'], :], self.linear.bias[:constant.ANSWER_NUM_DICT['kb']])
      #return F.linear(inputs, self.linear.weight[:constant.ANSWER_NUM_DICT['kb'], :])
    else:
      raise ValueError('Decoder error: output type not one of the valid')


class SimpleDecoder(nn.Module):
  def __init__(self, output_dim, answer_num):
    super(SimpleDecoder, self).__init__()
    self.answer_num = answer_num
    self.linear = nn.Linear(output_dim, answer_num, bias=False)

  def forward(self, inputs, output_type=None):
    output_embed = self.linear(inputs)
    return output_embed


class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1d = nn.Conv1d(100, 50, 5)  # input, output, filter_number
    self.char_W = nn.Embedding(115, 100)

  def forward(self, span_chars):
    char_embed = self.char_W(span_chars).transpose(1, 2)  # [batch_size, char_embedding, max_char_seq]
    conv_output = [self.conv1d(char_embed)]  # list of [batch_size, filter_dim, max_char_seq, filter_number]
    conv_output = [F.relu(c) for c in conv_output]  # batch_size, filter_dim, max_char_seq, filter_num
    cnn_rep = [F.max_pool1d(i, i.size(2)) for i in conv_output]  # batch_size, filter_dim, 1, filter_num
    cnn_output = torch.squeeze(torch.cat(cnn_rep, 1), 2)  # batch_size, filter_num * filter_dim, 1
    return cnn_output


class SelfAttentiveSum(nn.Module):
  """
  Attention mechanism to get a weighted sum of RNN output sequence to a single RNN output dimension.
  """
  def __init__(self, output_dim, hidden_dim):
    super(SelfAttentiveSum, self).__init__()
    self.key_maker = nn.Linear(output_dim, hidden_dim, bias=False)
    self.key_rel = nn.ReLU()
    self.hidden_dim = hidden_dim
    self.key_output = nn.Linear(hidden_dim, 1, bias=False)
    self.key_softmax = nn.Softmax()

  def _masked_softmax(self, X, mask=None, alpha=1e-20):
    # X, (batch_size, seq_length)
    X_max = torch.max(X, dim=1, keepdim=True)[0]
    X_exp = torch.exp(X - X_max)
    if mask is None:
      mask = (X != 0).float()
    X_exp = X_exp * mask
    X_softmax = X_exp / (torch.sum(X_exp,dim=1,keepdim=True) + alpha)
    return X_softmax

  def forward(self, input_embed):
    mask = (input_embed[:,:,0] != 0).float()
    input_embed_squeezed = input_embed.view(-1, input_embed.size()[2])
    k_d = self.key_maker(input_embed_squeezed)
    k_d = self.key_rel(k_d) # this leads all zeros
    if self.hidden_dim == 1:
      k = k_d.view(input_embed.size()[0], -1)
    else:
      k = self.key_output(k_d).view(input_embed.size()[0], -1)  # (batch_size, seq_length)
    weighted_keys = self._masked_softmax(k, mask=mask).view(input_embed.size()[0], -1, 1)
    #weighted_keys = self.key_softmax(k).view(input_embed.size()[0], -1, 1)
    weighted_values = torch.sum(weighted_keys * input_embed, 1)  # batch_size, seq_length, embed_dim
    return weighted_values, weighted_keys


class _Loss(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class BCEWithLogitsLossCustom(_Loss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='elementwise_mean', pos_weight=None, neg_weight=None):
        super(BCEWithLogitsLossCustom, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)
        self.register_buffer('neg_weight', neg_weight)
    
    def forward(self, input, target, pos_weight=None, neg_weight=None):
        if pos_weight is None:
            pos_weight = self.pos_weight
        if neg_weight is None:
            neg_weight = self.neg_weight
        return binary_cross_entropy_with_logits_custom(input, target,
                                                  self.weight,
                                                  pos_weight=pos_weight,
                                                  neg_weight=neg_weight,
                                                  reduction=self.reduction)

def binary_cross_entropy_with_logits_custom(input, target, weight=None, size_average=None,
                                     reduce=None, reduction='elementwise_mean', pos_weight=None, neg_weight=None):
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))
        
    max_val = (-input).clamp(min=0)
    
    log_sigmoid = nn.LogSigmoid()

    if pos_weight is None and neg_weight is None: # no positive/negative weight
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    elif pos_weight is not None and neg_weight is not None: # both positive/negative weight
        log_weight = neg_weight + (pos_weight - neg_weight) * target
        loss = neg_weight * input - neg_weight * input * target + log_weight * (max_val + ((-max_val).exp() + (-input - max_val).exp()).log())
        #loss = - pos_weight * target * log_sigmoid(input) - neg_weight * (1 - target) * log_sigmoid(-input)
    elif pos_weight is not None: # only positive weight
        log_weight = 1 + (pos_weight - 1) * target
        loss = input - input * target + log_weight * (max_val + ((-max_val).exp() + (-input - max_val).exp()).log())
    else: # only negative weight
        log_weight = neg_weight + (1 - neg_weight) * target
        loss = neg_weight * input - neg_weight * input * target + log_weight * (max_val + ((-max_val).exp() + (-input - max_val).exp()).log())
        #loss = - target * log_sigmoid(input) - neg_weight * (1 - target) * log_sigmoid(-input)
        
    if weight is not None:
        loss = loss * weight

    if reduction == 'none':
        return loss
    elif reduction == 'elementwise_mean':
        return loss.mean()
    else:
        return loss.sum()


class StructuredPerceptronLoss(_Loss):
  def __init__(self, size_average=None, reduce=None, reduction='elementwise_mean'):
    super(StructuredPerceptronLoss, self).__init__(size_average, reduce, reduction)

  def forward(self, input, target):

    loss = torch.clamp(input - target, min=0.)

    if self.reduction == 'none':
        return loss
    elif self.reduction == 'elementwise_mean':
        return loss.mean()
    else:
        return loss.sum()


class ELMoWeightedSum(nn.Module):
  def __init__(self):
    super(ELMoWeightedSum, self).__init__()
    self.gamma = nn.Parameter(torch.randn(1)) 
    self.S = nn.Parameter(torch.randn(1, 3))
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    """
    x:  ELMo vectors of (batch size, 3, 1024) or (batch size, 3, seq len, 1024).
    """
    S = self.softmax(self.S) # normalize
    if x.dim() == 3:
      batch_size, n_layers, emb_dim = x.shape
      x = x.permute(0, 2, 1).contiguous().view(-1, 3) # (batch_size*1024, 3)
      x = (x * S).sum(1) * self.gamma # (batch_size*1024, 1)
      x = x.view(batch_size, emb_dim) # (batch_size, 1024)
    elif x.dim() == 4:
      batch_size, n_layers, seq_len, emb_dim = x.shape
      x = x.permute(0, 2, 3, 1).contiguous().view(-1, 3) # (batch_size*seq_len*1024, 3)
      x = (x * S).sum(1) * self.gamma # (batch_size*seq_len*1024, 1)
      x = x.view(batch_size, seq_len, emb_dim) # (batch_size, seq_len, 1024)
    else:
      print('Wrong input dimension: x.dim() = ' + repr(x.dim()))
      raise ValueError
    return x


class LinearProjection(nn.Module):
  def __init__(self, in_features, out_features, bias=True):
    super(LinearProjection, self).__init__()
    self.in_features = in_features
    self.out_features = out_features 
    self.linear = nn.Linear(in_features, out_features, bias=True)

  def forward(self, x):
    """
    x:  ELMo vectors of (batch size, 3, 1024) or (batch size, 3, seq len, 1024).
    """
    if x.dim() == 2:
      batch_size, emb_dim = x.shape
      x = x.view(-1, self.in_features) # (batch_size, 1024)
      x = self.linear(x)
      x = x.view(batch_size, self.out_features) # (batch_size, 300)
    elif x.dim() == 3:
      batch_size, seq_len, emb_dim = x.shape
      x = x.view(-1, self.in_features) # (batch_size*seq_len, 300)
      x = self.linear(x)
      x = x.view(batch_size, seq_len, self.out_features) # (batch_size, seq_len, 300)
    else:
      print('Wrong input dimension: x.dim() = ' + repr(x.dim()))
      raise ValueError
    return x