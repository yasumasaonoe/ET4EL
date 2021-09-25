from os import path
from typing import List, Tuple

import pytorch_lightning as pl
import torch
import torchmetrics
from torch.nn import functional as F

from et4el.embedder import ELMO_EMBEDDINGS_DIM, ELMoPretrainedEmbedder
from et4el.encoder import MentionEncoder, SentenceEncoder, SimpleDecoder
from et4el.utils import MentionHandler, Mentions, load_vocab_dict

PATH_TO_VOCAB = path.normpath(path.join(path.dirname(__file__), "ontology/conll_categories.txt"))


class FineGrainedEntityTyper(pl.LightningModule):
    def __init__(self,
                 learning_rate=2e-3,
                 mention_dropout=0.5,
                 input_dropout=0.5,
                 rnn_dim=50,
                 cnn_dim=50,
                 mask_dim=50,
                 attention_dim=100,
                 answer_num=60000,
                 max_mention_length=10,
                 max_context_length=50,
                 threshold=0.5,
                 **_kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate

        self.answer_num = answer_num
        self.threshold = threshold

        embeddings_dim = ELMO_EMBEDDINGS_DIM
        output_dim = 2 * rnn_dim + embeddings_dim + cnn_dim

        self.pre_handler = MentionHandler(max_mention_length, max_context_length)
        self.embedder = ELMoPretrainedEmbedder(self.device)
        self.sentence_encoder = SentenceEncoder(input_dropout, rnn_dim, embeddings_dim, mask_dim, attention_dim)
        self.mention_encoder = MentionEncoder(mention_dropout, cnn_dim, embeddings_dim, attention_dim)
        self.decoder = SimpleDecoder(output_dim, answer_num)

        self.answer2id = load_vocab_dict(PATH_TO_VOCAB, vocab_max_size=self.answer_num)
        self.id2answer = {v: k for k, v in self.answer2id.items()}

        self.f1 = torchmetrics.F1(num_classes=answer_num, threshold=threshold, average="macro")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("FineGrainedEntityTyper")
        parser.add_argument("--mention-dropout", type=float, default=0.5)
        parser.add_argument("--input-dropout", type=float, default=0.5)
        parser.add_argument("--rnn-dim", type=int, default=50)
        parser.add_argument("--cnn-dim", type=int, default=50)
        parser.add_argument("--mask-dim", type=int, default=50)
        parser.add_argument("--attention-dim", type=int, default=100)
        parser.add_argument("--answer-num", type=int, default=60000)
        parser.add_argument("--max-mention-length", type=int, default=10)
        parser.add_argument("--max-context-length", type=int, default=50)
        parser.add_argument("--threshold", type=float, default=0.5)
        return parent_parser

    def to(self, *args, **kwargs):
        out = torch._C._nn._parse_to(*args, **kwargs)
        self.embedder.to(device=out[0])
        return super().to(*args, **kwargs)

    def _get_logits(self, mentions):
        """
        1. Get ELMo Embeddings
        2. Encode Sentence
        3. Encode Mention (word level and char level)
        4. Concat and decode all three vectors v = [s, m_word, m_char]
        """
        mentions = self.pre_handler.prepare_mentions(mentions)
        embeddings = self.embedder.embed(mentions)
        sequence_rep = self.sentence_encoder(mentions, embeddings)
        mention_rep = self.mention_encoder(mentions, embeddings)

        representation = torch.cat((sequence_rep, mention_rep), 1)
        logits = self.decoder(representation)
        return logits

    def forward(self, mentions: Mentions):
        logits = self._get_logits(mentions)
        outputs = torch.sigmoid(logits)

        # Decode predictions to categories
        predictions: List[List[Tuple[str, float]]] = []
        for output in outputs:
            output_indices = (output > self.threshold).nonzero().squeeze(1)
            if len(output_indices) == 0:
                output_indices = torch.argmax(output, dim=0, keepdim=True)
            predicted_categories = [(self.id2answer[i.item()], output[i].item()) for i in output_indices]
            predictions.append(predicted_categories)
        return predictions

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def training_step(self, batch, batch_idx):
        mentions, categories_batch = batch

        # Encode target categories to target tensor
        targets = torch.zeros([len(categories_batch), self.answer_num], device=self.device)
        for i, categories in enumerate(categories_batch):
            answer_ids = [self.answer2id[c] for c in categories if c in self.answer2id]
            for answer_idx in answer_ids:
                targets[i, answer_idx] = 1

        logits = self._get_logits(mentions)
        loss = F.binary_cross_entropy_with_logits(logits, targets)

        self.log("train_loss", loss)
        return loss

    def _eval_step(self, batch):
        mentions, categories_batch = batch

        # Encode target categories to target tensor
        targets = torch.zeros([len(categories_batch), self.answer_num], dtype=torch.int, device=self.device)
        for i, categories in enumerate(categories_batch):
            answer_ids = [self.answer2id[c] for c in categories if c in self.answer2id]
            for answer_idx in answer_ids:
                targets[i, answer_idx] = 1

        logits = self._get_logits(mentions)
        outputs = torch.sigmoid(logits)
        return outputs, targets

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch)

    def validation_epoch_end(self, results):
        # results = List[(outputs, targets) from each validation_step]
        outputs = torch.cat([outputs for outputs, targets in results])  # .permute(1, 0)
        targets = torch.cat([targets for outputs, targets in results])  # .permute(1, 0)
        accuracy = self.f1(outputs, targets)
        self.log("validation_accuracy", accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch)

    def test_epoch_end(self, results):
        # results = List[(outputs, targets) from each test_step]
        outputs = torch.cat([outputs for outputs, targets in results])  # .permute(1, 0)
        targets = torch.cat([targets for outputs, targets in results])  # .permute(1, 0)
        accuracy = self.f1(outputs, targets)
        self.log("test_accuracy", accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
