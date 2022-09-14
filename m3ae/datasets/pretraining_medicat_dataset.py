from .base_dataset import BaseDataset


class MedicatDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["medicat_train"]
        elif split == "val":
            names = ["medicat_val"]
        elif split == "test":
            names = ["medicat_test"]
        else:
            raise ValueError

        super().__init__(*args, **kwargs, names=names, text_column_name="caption")

    def __getitem__(self, index):
        return self.get_suite(index)
