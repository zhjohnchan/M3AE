import json
import os
import re
import random
import pandas as pd

from make_arrow import make_arrow, make_arrow_vqa, make_arrow_melinda


def prepro_vqa_vqa_rad():
    random.seed(42)

    data = {
        "train": [],
        "val": [],
        "test": []
    }

    data_root = "data/finetune_data/vqa_rad/"
    image_root = f"{data_root}/images"

    for split in ["train", "val", "test"]:
        with open(f"{data_root}/{split}set.json", "r") as fp:
            samples = json.load(fp)
            for sample in samples:
                img_path = os.path.join(image_root, sample["image_name"])
                qid = sample["qid"]
                question = sample["question"]
                answer = sample["answer"]
                answer_type = sample["answer_type"]
                data[split].append({
                    "img_path": img_path,
                    "qid": qid,
                    "question": question,
                    "answer": answer,
                    "answer_type": answer_type
                })
    make_arrow_vqa(data, "vqa_vqa_rad", "data/finetune_arrows/")


def prepro_vqa_slack():
    random.seed(42)

    data = {
        "train": [],
        "val": [],
        "test": []
    }

    data_root = "data/finetune_data/slack/"
    image_root = f"{data_root}/imgs"

    for split, file in zip(["train", "val", "test"], ["train.json", "validate.json", "test.json"]):
        with open(f"{data_root}/{file}", "r") as fp:
            samples = json.load(fp)
            for sample in samples:
                if sample["q_lang"] != "en":
                    continue
                img_path = os.path.join(image_root, sample["img_name"])
                qid = sample["qid"]
                question = sample["question"]
                answer = sample["answer"]
                answer_type = sample["answer_type"]
                data[split].append({
                    "img_path": img_path,
                    "qid": qid,
                    "question": question,
                    "answer": answer,
                    "answer_type": answer_type
                })
    make_arrow_vqa(data, "vqa_slack", "data/finetune_arrows/")


def prepro_vqa_medvqa2019():
    random.seed(42)

    data = {
        "train": [],
        "val": [],
        "test": []
    }

    data_root = "data/finetune_data/medvqa_2019/"
    image_root = "data/finetune_data/medvqa_2019/{}/images"

    offset = 0
    for split in ["train", "val", "test"]:
        samples = open(f"{data_root}/{split}/QA/Modality.csv").read().strip().split("\n") + \
                          open(f"{data_root}/{split}/QA/Organ.csv").read().strip().split("\n") + \
                          open(f"{data_root}/{split}/QA/Plane.csv").read().strip().split("\n")
        samples = [[idx + offset] + question.split("|") for idx, question in enumerate(samples)]
        offset += len(samples)
        for sample in samples:
            img_path = os.path.join(image_root.format(split), sample[1] + ".jpg")
            qid = sample[0]
            question = sample[2]
            answer = sample[3]
            answer_type = "OPEN"
            data[split].append({
                "img_path": img_path,
                "qid": qid,
                "question": question,
                "answer": answer,
                "answer_type": answer_type
            })
    make_arrow_vqa(data, "vqa_medvqa_2019", "data/finetune_arrows/")


def prepro_cls_melinda():
        random.seed(42)

        data = {
            "train": [],
            "val": [],
            "test": []
        }

        data_root = "data/finetune_data/melinda"
        image_root = f"{data_root}/melinda_images"

        for split, file in zip(["train", "val", "test"], ["train.csv", "dev.csv", "test.csv"]):
            samples = pd.read_csv(f"{data_root}/{file}")
            for sample_idx, sample in samples.iterrows():

                img_path = os.path.join(image_root, sample["figure_file"])
                texts = [sample["caption"]]
                i_meth = sample["i_meth"]
                p_meth = sample["p_meth"]
                i_meth_label = sample["i_meth_label"]
                p_meth_label = sample["p_meth_label"]

                if len(texts) > 0:
                    data[split].append({
                        "img_path": img_path,
                        "texts": texts,
                        "i_meth": i_meth,
                        "p_meth": p_meth,
                        "i_meth_label": i_meth_label,
                        "p_meth_label": p_meth_label
                    })

        make_arrow_melinda(data, "cls_melinda", "data/finetune_arrows/")


def prepro_irtr_roco(min_length=3):
    random.seed(42)

    data = {
        "train": [],
        "val": [],
        "test": []
    }
    roco_data_root = "data/pretrain_data/roco"
    roco_image_root = "data/pretrain_data/roco/{}/radiology/images/"

    for split in ["train", "val", "test"]:
        with open(f"{roco_data_root}/{split}/radiology/captions.txt", "r") as fp:
            lines = fp.read().strip().split("\n")
            random.shuffle(lines)
            for line_idx, line in enumerate(lines):
                str_splits = line.strip().split('\t')
                if len(str_splits) == 2:
                    img_path = os.path.join(roco_image_root.format(split), str_splits[0] + ".jpg")
                    texts = [str_splits[1]]
                    texts = [re.sub(r"\s+", " ", text) for text in texts]
                    texts = [text for text in texts if len(text.split()) >= min_length]
                    if len(texts) > 0:
                        data[split].append({
                            "img_path": img_path,
                            "texts": texts
                        })
                        if split == "val" and len(data[split]) == 2000:
                            break
                        if split == "test" and len(data[split]) == 2000:
                            break
    make_arrow(data, "irtr_roco", "data/finetune_arrows/")


if __name__ == '__main__':
    prepro_vqa_vqa_rad()
    prepro_vqa_slack()
    prepro_vqa_medvqa2019()
    prepro_cls_melinda()
    prepro_irtr_roco()
