from .base_datamodule import BaseDataModule
from ..datasets import ROCODataset


class ROCODataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return ROCODataset

    @property
    def dataset_cls_no_false(self):
        return ROCODataset

    @property
    def dataset_name(self):
        return "roco"
