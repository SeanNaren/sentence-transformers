import json
import os

from sentence_transformers.readers import NLIDataReader
from . import InputExample


class MedNLIDataReader(NLIDataReader):
    def __init__(self, dataset_folder):
        super().__init__(dataset_folder)

    def get_examples(self, filename, max_examples=0):
        """
        data_splits specified which data split to use (train, dev, test).
        Expects that self.dataset_folder contains the files mli_train_v1.jsonl, mli_dev_v1.jsonl, mli_test_v1.jsonl
        """
        raw_files = []
        with open(os.path.join(self.dataset_folder, filename)) as f:
            for line in f:
                raw_files.append(json.loads(line))

        examples = []
        for sample in raw_files:
            sentence_a, sentence_b, label = sample['sentence1'], sample['sentence2'], sample['gold_label']
            guid = sample['pairID']
            examples.append(InputExample(guid=guid, texts=[sentence_a, sentence_b], label=self.map_label(label)))

            if 0 < max_examples <= len(examples):
                break

        return examples
