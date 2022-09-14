from .base_datamodule import BaseDataModule
from ..datasets import CLSMELINDADataset


class CLSMELINDADataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return CLSMELINDADataset

    @property
    def dataset_cls_no_false(self):
        return CLSMELINDADataset

    @property
    def dataset_name(self):
        return "cls_melinda"
