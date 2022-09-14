from .base_datamodule import BaseDataModule
from ..datasets import IRTRROCODataset


class IRTRROCODataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return IRTRROCODataset

    @property
    def dataset_cls_no_false(self):
        return IRTRROCODataset

    @property
    def dataset_name(self):
        return "irtr_roco"
