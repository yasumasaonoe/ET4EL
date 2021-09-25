import torch
from allennlp.commands.elmo import ElmoEmbedder

from et4el.utils import EmbeddedBatch, Mentions

# ELMo
ELMO_OPTIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
ELMO_WEIGHT_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
ELMO_EMBEDDINGS_DIM = 1024


class ELMoPretrainedEmbedder():
    """ Get ELMo Embeddings
    """
    def __init__(self, device: torch.device):
        self.device = device
        cuda_device = -1 if self.device.type == "cpu" else self.device.index or 0
        self.embedder = ElmoEmbedder(ELMO_OPTIONS_FILE, ELMO_WEIGHT_FILE, cuda_device=cuda_device)

    def to(self, device: torch.device):
        self.device = device
        cuda_device = -1 if self.device.type == "cpu" else self.device.index or 0
        if cuda_device >= 0:
            self.embedder.elmo_bilm = self.embedder.elmo_bilm.cuda(device=cuda_device)
        self.embedder.cuda_device = cuda_device
        return self

    def embed(self, mentions: Mentions):
        """ Embed mentions to sentence- and mention embeddings.

        Args:
            mentions (Mentions): List of mentions

        Returns:
            EmbeddedBatch: Mention and sentence embeddings (grouped in own datatype wrapper)

        Dims:
            - EmbeddedBatch.mention_embeddings.shape = `(batch_size, 3, max_mention_tokens_length, ELMO_EMBEDDINGS_DIM)`
            - EmbeddedBatch.sentence_embeddings.shape = `(batch_size, 3, max_tokens_length, ELMO_EMBEDDINGS_DIM)`

            with:
            - `batch_size = Number of passed mentions`,
            - `max_tokens_length = Length of the longest mention sentence`,
            - `max_mention_tokens_length = Length of the longest mention` and
            - `ELMO_EMBEDDINGS_DIM = Output dim from ELMo embedder, defined by loaded options and weights`
        """
        bsz = len(mentions)
        embs = self.embedder.embed_batch(mentions.tokens)

        # Sentence Embeddings
        max_tokens_length = max(mentions.tokens_lengths)
        sentence_embeddings = torch.zeros([bsz, 3, max_tokens_length, ELMO_EMBEDDINGS_DIM], device=self.device)
        for i, emb in enumerate(embs):
            _, token_length, _ = emb.shape
            sentence_embeddings[i, :, :token_length, :] = torch.from_numpy(emb)

        # Mention Embeddings
        max_mention_tokens_length = max([mention.mention.count(" ") + 1 for mention in mentions])
        mention_embeddings = torch.zeros([bsz, 3, max_mention_tokens_length, ELMO_EMBEDDINGS_DIM], device=self.device)
        for i, (emb, mention) in enumerate(zip(embs, mentions)):
            start_ind, end_ind = mention.borders
            mention_length = end_ind - start_ind
            mention_embeddings[i, :, :mention_length, :] = torch.from_numpy(emb[:, start_ind:end_ind, :])

        return EmbeddedBatch(mention_embeddings, sentence_embeddings)
