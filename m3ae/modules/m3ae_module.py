import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel
from transformers.models.bert.modeling_bert import BertConfig, BertModel

from m3ae.modules import objectives, m3ae_utils
from m3ae.modules import prediction_heads
from m3ae.modules.language_encoders.bert_model import BertCrossLayer
from m3ae.modules.m3ae_utils import init_weights
from m3ae.modules.vision_encoders import swin_transformer as swin
from m3ae.modules.vision_encoders.clip_model import build_model, adapt_position_encoding
from m3ae.modules.vision_encoders.swin_helpers import swin_adapt_position_encoding


class M3AETransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        # == Begin: 1. Build Models ==
        self.is_clip = ('swin' not in config['vit'])
        if 'roberta' in config['tokenizer']:
            bert_config = RobertaConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )
        elif 'bert' in config['tokenizer']:
            bert_config = BertConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )
        else:
            raise ValueError

        resolution_after = config['image_size']
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                if self.is_clip:
                    build_model(config['vit'], resolution_after=resolution_after)
                else:
                    getattr(swin, self.hparams.config["vit"])(pretrained=True, config=self.hparams.config)
                if 'roberta' in config['tokenizer']:
                    RobertaModel.from_pretrained(config['tokenizer'])
                else:
                    BertModel.from_pretrained(config['tokenizer'])
            torch.distributed.barrier()
        if self.is_clip:
            self.vision_encoder = build_model(config['vit'], resolution_after=resolution_after)
        else:
            self.vision_encoder = getattr(swin, self.hparams.config["vit"])(pretrained=True, config=self.hparams.config)
            self.vision_pooler = nn.AdaptiveAvgPool1d(1)
        if 'roberta' in config['tokenizer']:
            self.language_encoder = RobertaModel.from_pretrained(config['tokenizer'])
        else:
            self.language_encoder = BertModel.from_pretrained(config['tokenizer'])

        self.multi_modal_language_proj = nn.Linear(config['input_text_embed_size'], config['hidden_size'])
        self.multi_modal_language_proj.apply(init_weights)
        self.multi_modal_vision_proj = nn.Linear(config['input_image_embed_size'], config['hidden_size'])
        self.multi_modal_vision_proj.apply(init_weights)

        self.modality_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.modality_type_embeddings.apply(init_weights)

        self.multi_modal_vision_layers = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(config['num_top_layer'])])
        self.multi_modal_vision_layers.apply(init_weights)
        self.multi_modal_language_layers = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(config['num_top_layer'])])
        self.multi_modal_language_layers.apply(init_weights)

        self.multi_modal_vision_pooler = prediction_heads.Pooler(config["hidden_size"])
        self.multi_modal_vision_pooler.apply(init_weights)
        self.multi_modal_language_pooler = prediction_heads.Pooler(config["hidden_size"])
        self.multi_modal_language_pooler.apply(init_weights)
        # == End  : 1. Build Models ==

        # == Begin: 2. Build Pre-Training Heads ==
        if config["loss_names"]["mlm"] > 0:
            self.mlm_head = prediction_heads.MLMHead(bert_config)
            self.mlm_head.apply(init_weights)
        if config["loss_names"]["mim"] > 0:
            self.mim_head = prediction_heads.MIMHead(config)
            self.mim_head.apply(init_weights)
        if config["loss_names"]["itm"] > 0 or self.hparams.config["loss_names"]["irtr"] > 0:
            self.itm_head = prediction_heads.ITMHead(config["hidden_size"] * 2)
            self.itm_head.apply(init_weights)
        # == End  : 2. Build Pre-Training Heads ==

        # == Begin: 3. Load Models ==
        if self.hparams.config["load_path"] != "" and not self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            if self.is_clip:
                state_dict = adapt_position_encoding(state_dict,
                                                     after=resolution_after,
                                                     patch_size=self.hparams.config['patch_size'])
            else:
                state_dict = swin_adapt_position_encoding(state_dict, after=resolution_after)
            self.load_state_dict(state_dict, strict=False)
        # == End  : 3. Load Models ==

        # == 4. Build Heads For Downstream Tasks ==
        hs = self.hparams.config["hidden_size"]
        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqa_label_size"]
            self.vqa_head = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_head.apply(init_weights)

        if self.hparams.config["loss_names"]["cls"] > 0:
            ms = self.hparams.config["melinda_label_size"][self.hparams.config["label_column_name"]]
            self.cls_head = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, ms),
            )
            self.cls_head.apply(init_weights)

        if self.hparams.config["loss_names"]["irtr"] > 0:
            self.irtr_head = nn.Linear(hs * 2, 1)
            self.irtr_head.weight.data = self.itm_head.fc.weight.data[1:, :]
            self.irtr_head.bias.data = self.itm_head.fc.bias.data[1:]
            self.margin = 0.2
            for p in self.itm_head.parameters():
                p.requires_grad = False

        m3ae_utils.set_metrics(self)
        self.current_tasks = list()
        # == End:  4. Build Heads For Downstream Tasks ==

        # == Begin: 5. Load Models For Testing ==
        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            state_dict = adapt_position_encoding(state_dict, after=resolution_after,
                                                 patch_size=self.hparams.config['patch_size'])
            self.load_state_dict(state_dict, strict=False)
        # == End  : 5. Load Models For Testing ==

    def random_masking(self, x, mask_ratio):
        x_ = x[:, :1]
        x = x[:, 1:]
        pos_embed = self.vision_encoder.visual.positional_embedding.unsqueeze(0).to(x)

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x += pos_embed[:, 1:]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # append cls token
        x_ = x_ + pos_embed[:, :1]
        x_masked = torch.cat((x_, x_masked), dim=1)

        return x_masked, mask, ids_restore

    def patchify(self, imgs):
        p = self.hparams.config["patch_size"]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        p = self.hparams.config["patch_size"]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def infer(
            self,
            batch,
            mask_text=False,
            mask_image=False,
            image_token_type_idx=1,
            img=None,
            output_attentions=False,
            unimodal=False
    ):
        ret = dict()

        # == Begin: Fetch the inputs ==
        if img is None:
            if f"image_{image_token_type_idx - 1}" in batch:
                img_key = f"image_{image_token_type_idx - 1}"
            else:
                img_key = "image"
            img = batch[img_key][0]
        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        device = text_ids.device
        # == End  : Fetch the inputs ==

        # == Begin: Text Encoding ==
        uni_modal_text_feats = self.language_encoder.embeddings(input_ids=text_ids)
        text_input_shape = text_masks.size()
        extended_text_masks = self.language_encoder.get_extended_attention_mask(text_masks, text_input_shape, device)
        for layer in self.language_encoder.encoder.layer:
            uni_modal_text_feats = layer(uni_modal_text_feats, extended_text_masks)[0]
        uni_modal_text_feats = self.multi_modal_language_proj(uni_modal_text_feats)
        # == End  : Text Encoding ==

        # == Begin: Image Encoding ==
        if mask_image:
            # == Begin: Image Masking ==
            # Mask: length -> length * mask_ratio
            # Perform position embedding inside the masking function
            uni_modal_image_feats = self.vision_encoder.forward_patch_embed(img)
            uni_modal_image_feats, mim_masks, mim_ids_restore = self.random_masking(uni_modal_image_feats,
                                                                                    self.hparams.config["mim_prob"])
            uni_modal_image_feats = self.vision_encoder.forward_trans(uni_modal_image_feats)
            ret["mim_masks"] = mim_masks
            ret["mim_ids_restore"] = mim_ids_restore
            # == End  : Image Masking ==
        else:
            uni_modal_image_feats = self.vision_encoder(img)
        uni_modal_image_feats = self.multi_modal_vision_proj(uni_modal_image_feats)
        image_masks = torch.ones((uni_modal_image_feats.size(0), uni_modal_image_feats.size(1)), dtype=torch.long,
                                 device=device)
        extended_image_masks = self.language_encoder.get_extended_attention_mask(image_masks, image_masks.size(),
                                                                                 device)
        # == End  : Image Encoding ==

        # == Begin: Assign Type Embeddings ==
        uni_modal_text_feats, uni_modal_image_feats = (
            uni_modal_text_feats + self.modality_type_embeddings(torch.zeros_like(text_masks)),
            uni_modal_image_feats + self.modality_type_embeddings(torch.full_like(image_masks, image_token_type_idx)),
        )
        # == End  : Assign Type Embeddings ==

        # == Begin: Multi-Modal Fusion ==
        ret["attentions"] = {"text2image_attns": [], "image2text_attns": []} if output_attentions else None
        x, y = uni_modal_text_feats, uni_modal_image_feats
        for layer_idx, (text_layer, image_layer) in enumerate(zip(self.multi_modal_language_layers,
                                                                  self.multi_modal_vision_layers)):
            # == Begin: Fetch the intermediate outputs (different layers to perform MIM) ==
            if mask_image and self.hparams.config["mim_layer"] == layer_idx:
                ret[f"multi_modal_text_feats_{layer_idx}"], ret[f"multi_modal_image_feats_{layer_idx}"] = x, y
            # == End  : Fetch the intermediate outputs (different layers to perform MIM) ==
            # == Begin: Co-Attention ==
            x1 = text_layer(x, y, extended_text_masks, extended_image_masks, output_attentions=True)
            y1 = image_layer(y, x, extended_image_masks, extended_text_masks, output_attentions=True)
            x, y = x1[0], y1[0]
            # == End: Co-Attention ==
            # == Begin: For visualization: Return the attention weights ==
            if output_attentions:
                ret["attentions"]["text2image_attns"].append(x1[1:])
                ret["attentions"]["image2text_attns"].append(y1[1:])
            # == End  : For visualization: Return the attention weights ==
        # == End  : Multi-Modal Fusion ==

        # == Begin: == Output Multi-Modal Features ==
        multi_modal_text_feats, multi_modal_image_feats = x, y
        multi_modal_text_cls_feats = self.multi_modal_language_pooler(x)
        if self.is_clip:
            multi_modal_image_cls_feats = self.multi_modal_vision_pooler(y)
        else:
            avg_image_feats = self.vision_pooler(multi_modal_image_feats.transpose(1, 2)).view(
                multi_modal_image_feats.size(0), 1, -1)
            multi_modal_image_cls_feats = self.multi_modal_vision_pooler(avg_image_feats)
        multi_modal_cls_feats = torch.cat([multi_modal_text_cls_feats, multi_modal_image_cls_feats], dim=-1)
        # == End  : == Output Multi-Modal Features ==

        ret.update({
            "images": img,
            "patched_images": self.patchify(img),
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "extended_image_masks": extended_image_masks,
            "extended_text_masks": extended_text_masks,
            "multi_modal_text_feats": multi_modal_text_feats,
            "multi_modal_image_feats": multi_modal_image_feats,
            "multi_modal_cls_feats": multi_modal_cls_feats,
        })

        return ret

    def forward(self, batch, test=False):
        ret = dict()

        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Pre-Training: Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Pre-Training: Masked Image Modeling
        if "mim" in self.current_tasks:
            ret.update(objectives.compute_mim(self, batch))

        # Pre-Training: Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm(self, batch))

        # Fine-Tuning: Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch, test=test))

        # Fine-Tuning: Image-Text Classification
        if "cls" in self.current_tasks:
            ret.update(objectives.compute_cls(self, batch, test=test))

        # Fine-Tuning: Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch, test))

        return ret

    def training_step(self, batch, batch_idx):
        m3ae_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v * self.hparams.config["loss_names"][k.replace("_loss", "")]
                          for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        m3ae_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        m3ae_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        m3ae_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        m3ae_utils.set_task(self)
        output = self(batch, test=True)

    def test_epoch_end(self, outs):
        m3ae_utils.epoch_wrapup(self, test=True)

    def configure_optimizers(self):
        return m3ae_utils.set_schedule(self)
