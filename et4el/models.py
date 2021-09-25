import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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
        S = self.softmax(self.S)  # normalize
        if x.dim() == 3:
            batch_size, n_layers, emb_dim = x.shape
            x = x.permute(0, 2, 1).contiguous().view(-1, 3)  # (batch_size*1024, 3)
            x = (x * S).sum(1) * self.gamma  # (batch_size*1024, 1)
            x = x.view(batch_size, emb_dim)  # (batch_size, 1024)
        elif x.dim() == 4:
            batch_size, n_layers, seq_len, emb_dim = x.shape
            x = x.permute(0, 2, 3, 1).contiguous().view(-1, 3)  # (batch_size*seq_len*1024, 3)
            x = (x * S).sum(1) * self.gamma  # (batch_size*seq_len*1024, 1)
            x = x.view(batch_size, seq_len, emb_dim)  # (batch_size, seq_len, 1024)
        else:
            print('Wrong input dimension: x.dim() = ' + repr(x.dim()))
            raise ValueError
        return x


class BiLSTM(nn.Module):
    def __init__(self, embeddings_dim, rnn_dim):
        super().__init__()
        self.lstm = nn.LSTM(embeddings_dim, rnn_dim, bidirectional=True, batch_first=True)

    def forward(self, weighted_embeddings, sequence_lengths):
        sorted_inputs, sorted_sequence_lengths, restoration_indices = sort_batch_by_length(
            weighted_embeddings, sequence_lengths)
        packed_sequence_input = pack_padded_sequence(sorted_inputs,
                                                     sorted_sequence_lengths.data.long().tolist(),
                                                     batch_first=True)
        packed_sequence_output, _ = self.lstm(packed_sequence_input, None)
        unpacked_sequence_tensor, _ = pad_packed_sequence(packed_sequence_output, batch_first=True)
        context_rep = unpacked_sequence_tensor.index_select(0, restoration_indices)
        return context_rep


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
        X_softmax = X_exp / (torch.sum(X_exp, dim=1, keepdim=True) + alpha)
        return X_softmax

    def forward(self, input_embed):
        mask = (input_embed[:, :, 0] != 0).float()
        input_embed_squeezed = input_embed.view(-1, input_embed.size()[2])
        k_d = self.key_maker(input_embed_squeezed)
        k_d = self.key_rel(k_d)  # this leads all zeros
        if self.hidden_dim == 1:
            k = k_d.view(input_embed.size()[0], -1)
        else:
            k = self.key_output(k_d).view(input_embed.size()[0], -1)  # (batch_size, seq_length)
        weighted_keys = self._masked_softmax(k, mask=mask).view(input_embed.size()[0], -1, 1)
        #weighted_keys = self.key_softmax(k).view(input_embed.size()[0], -1, 1)
        weighted_values = torch.sum(weighted_keys * input_embed, 1)  # batch_size, seq_length, embed_dim
        return weighted_values


class CNN(nn.Module):
    def __init__(self, output_dim):
        super(CNN, self).__init__()
        self.conv1d = nn.Conv1d(100, output_dim, 5)  # input, output, filter_number
        self.char_W = nn.Embedding(115, 100)

    def forward(self, span_chars):
        char_embed = self.char_W(span_chars).transpose(1, 2)  # [batch_size, char_embedding, max_char_seq]
        conv_output = [self.conv1d(char_embed)]  # list of [batch_size, filter_dim, max_char_seq, filter_number]
        conv_output = [F.relu(c) for c in conv_output]  # batch_size, filter_dim, max_char_seq, filter_num
        cnn_rep = [F.max_pool1d(i, i.size(2)) for i in conv_output]  # batch_size, filter_dim, 1, filter_num
        cnn_output = torch.squeeze(torch.cat(cnn_rep, 1), 2)  # batch_size, filter_num * filter_dim, 1
        return cnn_output
