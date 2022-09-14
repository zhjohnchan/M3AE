import json
import os
import random
import re

from make_arrow import make_arrow


def prepro_medicat(min_length=3):
    random.seed(42)

    data = {
        "train": [],
        "val": [],
        "test": []
    }

    data_root = "data/pretrain_data/medicat"
    image_root = f"{data_root}/release/figures/"
    medicat_ann_path = f"{data_root}/release/s2_full_figures_oa_nonroco_combined_medical_top4_public.jsonl"

    medicat_samples = [json.loads(sample) for sample in open(medicat_ann_path).read().strip().split("\n")]
    medicat_samples = [sample for sample in medicat_samples if sample["radiology"]]
    indices = list(range(len(medicat_samples)))
    random.shuffle(indices)
    splits = {
        "train": indices[:-2000],
        "val": indices[-2000:-1000],
        "test": indices[-1000:],
    }
    for split, split_indices in splits.items():
        for sample_idx in split_indices:
            sample = medicat_samples[sample_idx]
            img_path = os.path.join(image_root, sample["pdf_hash"] + "_" + sample["fig_uri"])
            texts = []
            if "s2_caption" in sample and len(sample["s2_caption"]) > 0:
                texts.append(sample["s2_caption"])
            if "s2orc_references" in sample and sample["s2orc_references"] is not None and len(
                    sample["s2orc_references"]) > 0:
                texts.extend(sample["s2orc_references"])
            texts = [re.sub(r"\s+", " ", text.strip()) for text in texts]
            texts = [text for text in texts if len(text.split()) >= min_length]
            if len(texts) > 0:
                data[split].append({
                    "img_path": img_path,
                    "texts": texts
                })

    make_arrow(data, "medicat", "data/pretrain_arrows/")


def prepro_roco(min_length=3):
    random.seed(42)

    data = {
        "train": [],
        "val": [],
        "test": []
    }
    roco_data_root = "data/pretrain_data/roco"
    roco_image_root = "data/pretrain_data/roco/{}/radiology/images/"
    medicat_roco_data_root = "data/pretrain_data/medicat"
    medicat_roco_paths = {
        "train": f"{medicat_roco_data_root}/net/nfs2.corp/allennlp/sanjays/roco_files/roco_train_references.jsonl",
        "val": f"{medicat_roco_data_root}/net/nfs2.corp/allennlp/sanjays/roco_files/roco_val_references.jsonl",
        "test": f"{medicat_roco_data_root}/net/nfs2.corp/allennlp/sanjays/roco_files/roco_test_references.jsonl"
    }

    medicat2roco = {}
    for split in ["train", "val", "test"]:
        with open(f"{roco_data_root}/{split}/radiology/dlinks.txt", "r") as fp:
            for line in fp:
                str_splits = line.strip().split('\t')
                medicat2roco[str_splits[1].split(' ')[2].split('/')[-1].split('.')[0] + "_" + str_splits[-1]] = \
                str_splits[0]

    for split, path in medicat_roco_paths.items():
        samples = [json.loads(sample) for sample in open(path).read().strip().split("\n")]
        for sample in samples:
            img_path = os.path.join(roco_image_root.format(split), medicat2roco[sample["roco_image_id"]] + ".jpg")
            texts = []
            if "gorc_references" in sample and sample["gorc_references"] is not None and len(
                    sample["gorc_references"]) > 0:
                texts.extend(sample["gorc_references"])
            texts = [re.sub(r"\s+", " ", text.strip()) for text in texts]
            texts = [text for text in texts if len(text.split()) >= min_length]
            if len(texts) > 0:
                data[split].append({
                    "img_path": img_path,
                    "texts": texts
                })

    for split in ["train", "val", "test"]:
        with open(f"{roco_data_root}/{split}/radiology/captions.txt", "r") as fp:
            for line in fp:
                str_splits = line.strip().split('\t')
                if len(str_splits) == 2:
                    img_path = os.path.join(roco_image_root.format(split), str_splits[0] + ".jpg")
                    texts = [str_splits[1]]
                    texts = [re.sub(r"\s+", " ", text.strip()) for text in texts]
                    texts = [text for text in texts if len(text.split()) >= min_length]
                    if len(texts) > 0:
                        data[split].append({
                            "img_path": img_path,
                            "texts": texts
                        })
    make_arrow(data, "roco", "data/pretrain_arrows/")


if __name__ == '__main__':
    prepro_medicat()
    prepro_roco()
