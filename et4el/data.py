import fnmatch
import glob
import json
import tarfile
from os import path

import pytorch_lightning as pl
from google_drive_downloader import GoogleDriveDownloader as gdd
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import IterableDataset

from et4el.utils import Mention

DATA_DIR = path.normpath(path.join(path.dirname(__file__), "../data/"))


def collate_fn(batch):
    x = [item[0] for item in batch]
    y = [item[1] for item in batch]
    return x, y


class EtConllSet(IterableDataset):
    """Dataset based on the CONLL Corpus, generated from the ET4EL team
    """
    def __init__(self, data_dir, file_matching, unzipped=False):
        """Dataset based ont he CONLL Corpus, generated from the ET4EL team

        Args:
            data_dir (str): Path to data directory
            file_matching (str): UNIX style based patter to match files against
            unzipped (bool, optional): Whether to read from unzipped folder or directly from compressed file. Defaults to False.
        """
        super().__init__()
        self.length = -1
        self.data_dir = data_dir
        self.file_matching = file_matching
        self.data_path = path.join(data_dir, file_matching)
        self.unzipped = unzipped

    def count_lines(self):
        """Iterable datasets don't provide a __len__ attribute by themself. Pre-calculating the length for optimisation.
        """
        if self.unzipped:
            self.length = sum(sum(1 for _ in open(shard_name)) for shard_name in glob.glob(self.data_path))
        else:
            t_file = path.join(self.data_dir, 'entity_typing_data.tar.gz')
            tar = tarfile.open(t_file, "r:gz")
            self.length = sum(
                sum(1 for _ in tar.extractfile(member)) for member in tar
                if fnmatch.fnmatch(member.name, self.file_matching))

    def __len__(self):
        return self.length

    def _data_step(self, line):
        """Reads examples from json line

        Returns:
            Tuple[Mention, List[str]]: Tuple of a Mention (own datatype, wrapping a mentions context etc.) and a list of target types.
        """
        example = json.loads(line.strip())
        mention = Mention(
            example["word"],
            example["left_context_text"],
            example["right_context_text"],
        )
        categories = example["y_category"]
        return mention, categories

    def __iter__(self):
        if self.unzipped:
            for shard_name in glob.glob(self.data_path):
                for line in open(shard_name):
                    yield self._data_step(line)
        else:
            t_file = path.join(self.data_dir, 'entity_typing_data.tar.gz')
            tar = tarfile.open(t_file, "r:gz")
            for member in tar:
                if not fnmatch.fnmatch(member.name, self.file_matching):
                    continue
                for line in tar.extractfile(member):
                    yield self._data_step(line)


class EntityTypingDataModule(pl.LightningDataModule):
    def __init__(self, data_dir=DATA_DIR, use_zip=False, batch_size=32, **_kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.unzip = not use_zip
        self.batch_size = batch_size

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("EntityTypingDataModule")
        parser.add_argument("--data-dir", type=str, default=DATA_DIR)
        parser.add_argument("--use-zip", action="store_true")
        parser.add_argument("--batch-size", type=int, default=32)
        return parent_parser

    def prepare_data(self):
        """Downloads and unzip data if not exists yet
        """
        if path.exists(path.join(self.data_dir, "entity_typing_data")):
            return

        t_file = path.join(self.data_dir, 'entity_typing_data.tar.gz')
        if not path.exists(t_file):
            gdd.download_file_from_google_drive(file_id='1m9CPaehSjlsFA6Na-bYZ2GWt_kzyfJTo',
                                                dest_path=t_file,
                                                showsize=True)
        if self.unzip:
            tar = tarfile.open(t_file, "r:gz")
            tar.extractall(self.data_dir)
            tar.close()

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.et_train = EtConllSet(self.data_dir, "entity_typing_data/train/et_conll_60k/train_*.json", self.unzip)
            self.et_valid = EtConllSet(self.data_dir, "entity_typing_data/validation/dev_et_conll_60k.json", self.unzip)
            self.et_train.count_lines()
            self.et_valid.count_lines()

        if stage == "test" or stage is None:
            self.et_test = EtConllSet(self.data_dir, "entity_typing_data/validation/dev_et_unseen_60k.json", self.unzip)
            self.et_test.count_lines()

    def train_dataloader(self):
        return DataLoader(self.et_train, batch_size=self.batch_size, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.et_valid, batch_size=self.batch_size, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.et_test, batch_size=self.batch_size, collate_fn=collate_fn)
