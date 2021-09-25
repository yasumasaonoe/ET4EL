from collections import defaultdict
from os import path

import torch
import torch.nn as nn

from et4el.models import CNN, BiLSTM, ELMoWeightedSum, SelfAttentiveSum
from et4el.utils import EmbeddedBatch, Mentions

PATH_TO_CHARDICT = path.normpath(path.join(path.dirname(__file__), "ontology/char_vocab.english.txt"))


class SentenceEncoder(nn.Module):
    """
        Encode Sentence
          - Get word lokation tokens
          - concat sentence embeddings with lokation tokens
          - fed into bi lstm (with dim_hid)
          - span attention
          - s = Attention(bi-LSTM([s'; l]))
    """
    def __init__(self, dropout_rate, rnn_dim, embeddings_dim, mask_dim, attention_dim):
        super().__init__()

        # Define dims
        self.rnn_dim = rnn_dim
        self.embeddings_dim = embeddings_dim  # Must be same as ELMo output dim (1024)
        self.mask_dim = mask_dim
        self.combined_dim = self.embeddings_dim + self.mask_dim
        self.attention_dim = attention_dim

        # Define networks
        self.weighted_sum = ELMoWeightedSum()
        self.location_LNN = nn.Linear(4, self.mask_dim)
        self.input_dropout = nn.Dropout(dropout_rate)
        self.bi_lstm = BiLSTM(self.combined_dim, self.rnn_dim)
        self.attentive_sum = SelfAttentiveSum(self.rnn_dim * 2, self.attention_dim)

    def get_location_tokens(self, mentions: Mentions, device) -> torch.Tensor:
        """Get location tokens:
        Each word is assigned one of four location tokens, based on whether
        - (1) the word is in the left context,
        - (2) the word is the first word of the mention span,
        - (3) the word is in the mention span (but not first), and
        - (4) the word is in the right context.

        Returns:
            Tensor: Location tokens. Dim: (batch_size, max_seq_length, 4)
        """
        max_seq_length = max(mentions.tokens_lengths)
        bsz = len(mentions)
        location_tokens = torch.zeros([bsz, max_seq_length, 4], device=device)
        for i, mention in enumerate(mentions):
            start_ind, end_ind = mention.borders

            location_tokens[i, :start_ind, 0] = 1.0
            location_tokens[i, start_ind, 1] = 1.0
            location_tokens[i, start_ind + 1:end_ind, 2] = 1.0
            location_tokens[i, end_ind:mention.tokens_length, 3] = 1.0
        return location_tokens

    def forward(self, mentions: Mentions, embeddings: EmbeddedBatch):
        """Get sentence representations.

        Args:
            mentions (Mentions): List of mentions
            embeddings (EmbeddedBatch): Embeddings from ELMo embedder

        Returns:
            Tensor: Sequence representation. Dim: (batch_size, 2*rnn_dim)
        """
        # embeddings.sentence_embeddings: torch.Size([bsz, 3, maximum_sentence_length, embeddings_dim])
        weighted_embeddings = self.weighted_sum(embeddings.sentence_embeddings)
        # weighted_embeddings: torch.Size([bsz, maximum_sentence_length, embeddings_dim])

        location_tokens = self.get_location_tokens(mentions, embeddings.sentence_embeddings.device).view(-1, 4)
        # location_tokens: torch.Size([bsz*maximum_sentence_length, 4])
        location_mask = self.location_LNN(location_tokens)
        # location_mask: torch.Size([bsz*maximum_sentence_length, mask_dim])
        location_mask = location_mask.view(weighted_embeddings.size()[0], -1, self.mask_dim)
        # location_mask: torch.Size([bsz, maximum_sentence_length, mask_dim])

        weighted_embeddings = torch.cat((weighted_embeddings, location_mask), 2)
        # weighted_embeddings: torch.Size([bsz, maximum_sentence_length, combined_dim])
        weighted_embeddings = self.input_dropout(weighted_embeddings)

        sequence_lengths = torch.tensor(mentions.tokens_lengths, device=embeddings.sentence_embeddings.device)

        sequence_rep = self.bi_lstm(weighted_embeddings, sequence_lengths)
        # sequence_rep: torch.Size([bsz, maximum_sentence_length, 2*rnn_dim])
        sequence_rep = self.attentive_sum(sequence_rep)
        # sequence_rep: torch.Size([bsz, 2*rnn_dim])
        return sequence_rep


class MentionEncoder(nn.Module):
    """
        Encode Mention (word level)
            - m' fed into bi lstm (with dim_hid)
            - concat hidden states of both directions
            - summed by span attention
            - m_word = Attention(bi-LSTM([m'; l]))
        Encode Mention (character level)
            - characters get embedded and fed into 1-D convolution
            - m_char = CNN(mention_chars)
    """
    def __init__(self, dropout_rate, cnn_dim, embeddings_dim, attention_dim):
        super().__init__()

        # Define dimensions
        self.cnn_dim = cnn_dim
        self.embeddings_dim = embeddings_dim  # Must be same as ELMo output dim (1024)
        self.attention_dim = attention_dim

        # Define networks
        self.weighted_sum = ELMoWeightedSum()
        self.input_dropout = nn.Dropout(dropout_rate)
        self.bi_lstm = BiLSTM(self.embeddings_dim, self.embeddings_dim // 2)
        self.attentive_sum = SelfAttentiveSum(self.embeddings_dim, self.attention_dim)
        self.cnn = CNN(self.cnn_dim)

        # Load char dictionary
        self.char_dict = defaultdict(int)
        char_vocab = [u"<unk>"]
        with open(PATH_TO_CHARDICT, encoding="utf-8") as f:
            char_vocab.extend(c.strip() for c in f.readlines())
            self.char_dict.update({c: i for i, c in enumerate(char_vocab)})

    @staticmethod
    def pad_slice(seq, seq_length, pad_token="<none>"):
        """Fills a sequence with a pad_token until it reached the desire length
        """
        return seq + ([pad_token] * (seq_length - len(seq)))

    def get_mention_characters(self, mentions: Mentions, device):
        """Gets characters from mention, padded to longest mention length
        """
        mentions_characters = [[self.char_dict[x] for x in list(mention.mention)] for mention in mentions]
        # max(..., 5): 5 because CNN uses 5 as kernel size in Conv1d
        max_span_chars = max(max(len(characters) for characters in mentions_characters), 5)
        mentions_characters = [
            self.pad_slice(characters, max_span_chars, pad_token=0) for characters in mentions_characters
        ]
        span_chars = torch.tensor(mentions_characters, dtype=torch.int64, device=device)
        # span_chars: torch.Size([bsz, max_span_chars])
        return span_chars

    def forward(self, mentions: Mentions, embeddings: EmbeddedBatch):
        # embeddings.mention_embeddings: torch.Size([bsz, 3, maximum_mention_length, embeddings_dim])
        weighted_embeddings = self.weighted_sum(embeddings.mention_embeddings)
        # weighted_embeddings: torch.Size([bsz, maximum_mention_length, embeddings_dim])
        weighted_embeddings = self.input_dropout(weighted_embeddings)

        mention_lengths = torch.tensor([mention.mention.count(" ") + 1 for mention in mentions],
                                       device=embeddings.mention_embeddings.device)

        mention_word = self.bi_lstm(weighted_embeddings, mention_lengths)
        # mention_word: torch.Size([bsz, maximum_mention_length, embeddings_dim])
        mention_word = self.attentive_sum(mention_word)
        # mention_word: torch.Size([bsz, embeddings_dim])

        mention_chars = self.get_mention_characters(mentions, mention_word.device)
        # mention_chars: torch.Size([bsz, maximum_mention_length])
        mention_chars = self.cnn(mention_chars)
        # mention_chars: torch.Size([bsz, cnn_dim])
        mention_rep = torch.cat((mention_word, mention_chars), 1)
        # mention_rep: torch.Size([bsz, embeddings_dim + cnn_dim])
        return mention_rep


class SimpleDecoder(nn.Module):
    def __init__(self, output_dim, answer_num):
        super().__init__()
        self.linear = nn.Linear(output_dim, answer_num, bias=False)

    def forward(self, inputs):
        output_embed = self.linear(inputs)
        return output_embed
