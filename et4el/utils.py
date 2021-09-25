from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple

import torch


def load_vocab_dict(vocab_file_name, vocab_max_size=None):
    """Loads vocabulary from file ("conll_categories.txt") and maps the first X entries to a dict of ids and categories
    """
    with open(vocab_file_name, encoding="utf-8") as f:
        text = [x.strip() for x in f.readlines()]
        text = text[:vocab_max_size]
        file_content = dict(zip(text, range(len(text))))
    return file_content


@dataclass(frozen=True)
class EmbeddedBatch():
    mention_embeddings: torch.Tensor
    sentence_embeddings: torch.Tensor


@dataclass(frozen=True)
class Mention():
    """Mention type which wraps a mention and its context into one single class

    Properties:
    `mention`: `str` The mention in the text
    `left_context`: `str` The context before the mention
    `right_context`: `str` The context after the mention
    `sequence`: `str` The complete sentence: left_context + mention + right_context
    `tokens`: `List[str]` Splitted sequence
    `tokens_length`: `int` Number of tokens in sequence
    `borders`: `Tuple[int, int]` Border indicies of the mention
    """

    mention: str
    left_context: str
    right_context: str

    def __repr__(self) -> str:
        return f"Mention({self.mention})"

    @staticmethod
    def find_index_in_sequence(a: list, b: list):
        return [(i, i + len(b)) for i in range(len(a)) if a[i:i + len(b)] == b]

    @property
    @lru_cache()
    def sequence(self) -> str:
        return f"{self.left_context} {self.mention} {self.right_context}"

    @property
    @lru_cache()
    def tokens(self) -> List[str]:
        return self.sequence.split()

    @property
    @lru_cache()
    def tokens_length(self) -> int:
        return self.sequence.count(" ") + 1

    @property
    @lru_cache()
    def borders(self) -> Tuple[int, int]:
        return self.find_index_in_sequence(self.tokens, self.mention.split())[0]


class Mentions(List[Mention]):
    """List of mentions, inherits from list

    """
    def __repr__(self) -> str:
        return f"Mentions({', '.join([m.mention for m in self])})"

    @property
    def sequences(self) -> List[str]:
        return [c.sequence for c in self]

    @property
    def tokens(self) -> List[List[str]]:
        return [c.tokens for c in self]

    @property
    def tokens_lengths(self) -> List[int]:
        return [c.tokens_length for c in self]


class MentionHandler():
    def __init__(self, max_mention_length, max_context_length):
        self.max_mention_length = max_mention_length
        self.max_context_length = max_context_length
        self.max_sequence_length = max_context_length + max_mention_length + max_context_length

    @staticmethod
    def replace_numbers(s):
        return ["<number>" if w.lower().isnumeric() else w for w in s]

    def trim_sequences(self, mention: Mention) -> Mention:
        """Trim sequences so they doesn't extend x tokens
        """
        start_idx, end_idx = mention.borders

        left_start_idx = max(start_idx - self.max_context_length, 0)
        right_end_idx = min(end_idx + self.max_context_length, self.max_sequence_length)

        left_seq = self.replace_numbers(mention.tokens[left_start_idx:start_idx])
        right_seq = self.replace_numbers(mention.tokens[end_idx:right_end_idx])

        mention_seq = self.replace_numbers(mention.mention.split()[:self.max_mention_length])
        mention = Mention(" ".join(mention_seq), " ".join(left_seq), " ".join(right_seq))
        return mention

    def prepare_mentions(self, mentions: Mentions) -> Mentions:
        prepared_mentions = Mentions()
        for mention in mentions:
            prepared_mentions.append(self.trim_sequences(mention))
        return prepared_mentions
