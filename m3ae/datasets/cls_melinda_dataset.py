import torch

from .base_dataset import BaseDataset


class CLSMELINDADataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["cls_melinda_train"]
        elif split == "val":
            names = ["cls_melinda_val"]
        elif split == "test":
            names = ["cls_melinda_test"]
        else:
            raise ValueError

        super().__init__(*args, **kwargs, names=names, text_column_name="caption")
        self.label_column_name = self.label_column_name
        self.labels = self.table[self.label_column_name].to_pandas().tolist()
        assert len(self.labels) == len(self.table)

    def __getitem__(self, index):
        return self.get_suite(index)

    def get_suite(self, index):
        ret = super(CLSMELINDADataset, self).get_suite(index)
        img_index, cap_index = self.index_mapper[index]
        ret["cls_labels"] = self.labels[img_index][cap_index]
        return ret

    def collate(self, batch, mlm_collator):
        dict_batch = super(CLSMELINDADataset, self).collate(batch, mlm_collator)

        dict_batch["cls_labels"] = torch.tensor([sample["cls_labels"] for sample in batch])
        return dict_batch
