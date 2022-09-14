import numpy as np
import sklearn.metrics as sklm
import torch
from pytorch_lightning.metrics import Metric


class Accuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )
        preds = logits.argmax(dim=-1)
        preds = preds[target != -100]
        target = target[target != -100]
        if target.numel() == 0:
            return 1

        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct / self.total


class Scalar(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)
        self.scalar += scalar
        self.total += 1

    def compute(self):
        return self.scalar / self.total


class VQAScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().float().to(self.score.device),
            target.detach().float().to(self.score.device),
        )
        logits = torch.max(logits, 1)[1]
        one_hots = torch.zeros(*target.size()).to(target)
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = one_hots * target

        self.score += scores.sum()
        self.total += len(logits)

    def compute(self):
        return self.score / self.total


class VQARADScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("close_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("close_total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("open_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("open_total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.best_score = 0
        self.best_close_score = 0
        self.best_open_score = 0

    def update(self, logits, target, types=None):
        logits, target = (
            logits.detach().float().to(self.score.device),
            target.detach().float().to(self.score.device),
        )
        logits = torch.max(logits, 1)[1]
        one_hots = torch.zeros(*target.size()).to(target)
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = one_hots * target

        close_scores = scores[types == 0]
        open_scores = scores[types == 1]

        self.close_score += close_scores.sum()
        self.close_total += len(close_scores)
        self.open_score += open_scores.sum()
        self.open_total += len(open_scores)

        self.score += scores.sum()
        self.total += len(scores)

    def compute(self):
        score = self.score / self.total
        return score

    def get_best_score(self):
        self.sync()
        score = self.score / self.total
        if score > self.best_score:
            self.best_score = score
            self.best_close_score = self.close_score / self.close_total if self.close_total != 0 else 0
            self.best_open_score = self.open_score / self.open_total if self.open_total != 0 else 0
        self.unsync()
        return self.best_score

    def get_best_close_score(self):
        return self.best_close_score

    def get_best_open_score(self):
        return self.best_open_score


class ROCScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("y_trues", default=[], dist_reduce_fx="cat")
        self.add_state("y_scores", default=[], dist_reduce_fx="cat")
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="mean")

    def update(self, logits, target):
        logits, target = (
            logits.detach().float(),
            target.detach().float(),
        )

        y_true = target
        y_score = 1 / (1 + torch.exp(-logits))
        self.y_trues.append(y_true)
        self.y_scores.append(y_score)

    def compute(self):
        try:
            score = sklm.roc_auc_score(np.concatenate([y_true.cpu().numpy() for y_true in self.y_trues], axis=0),
                                       np.concatenate([y_score.cpu().numpy() for y_score in self.y_scores], axis=0))
            self.score = torch.tensor(score).to(self.score)
        except ValueError:
            self.score = torch.tensor(0).to(self.score)
        return self.score


class F1Score(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("y_trues", default=[], dist_reduce_fx="cat")
        self.add_state("y_preds", default=[], dist_reduce_fx="cat")
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="mean")

    def update(self, logits, target):
        logits, target = (
            logits.detach().float(),
            target.detach().float(),
        )

        y_true = target
        y_score = 1 / (1 + torch.exp(-logits)) > 0.5
        self.y_trues.append(y_true)
        self.y_preds.append(y_score)

    def compute(self):
        try:
            score = sklm.f1_score(np.concatenate([y_true.cpu().numpy() for y_true in self.y_trues], axis=0),
                                  np.concatenate([y_pred.cpu().numpy() for y_pred in self.y_preds], axis=0))
            self.score = torch.tensor(score).to(self.score)
        except ValueError:
            self.score = torch.tensor(0).to(self.score)
        return self.score
