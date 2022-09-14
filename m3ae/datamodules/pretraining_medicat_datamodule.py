from .base_datamodule import BaseDataModule
from ..datasets import MedicatDataset


class MedicatDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return MedicatDataset

    @property
    def dataset_cls_no_false(self):
        return MedicatDataset

    @property
    def dataset_name(self):
        return "medicat"
