import torch
import torchmetrics
import wandb
from torchmetrics.segmentation import DiceScore


class Metrics:
    def __init__(self, device, mode="val"):
        self.mode = mode
        self.accuracy_metric = torchmetrics.Accuracy(task="binary").to(device)
        self.precision_metric = torchmetrics.Precision(task="binary").to(device)
        self.recall_metric = torchmetrics.Recall(task="binary").to(device)
        self.f1_metric = torchmetrics.classification.BinaryF1Score().to(device)
        if mode == "test":
            self.auroc = torchmetrics.classification.AUROC(task="binary").to(device)

    def batch_metrics(self, preds, labels):
        self.accuracy_metric.update(preds, labels)
        self.precision_metric.update(preds, labels)
        self.recall_metric.update(preds, labels)
        self.f1_metric.update(preds, labels)
        if self.mode == "test":
            self.auroc.update(preds, labels)

    def __reset(self):
        self.accuracy_metric.reset()
        self.precision_metric.reset()
        self.recall_metric.reset()
        self.f1_metric.reset()
        if self.mode == "test":
            self.auroc.reset()

    def epoch_metrics(self):
        acc = self.accuracy_metric.compute().item()
        prec = self.precision_metric.compute().item()
        rec = self.recall_metric.compute().item()
        f1 = self.f1_metric.compute().item()
        if self.mode == "test":
            auroc = self.auroc.compute().item()
        self.__reset()

        if self.mode == "test":
            return {"Accuracy":acc, "Precision":prec, "Recall":rec, "F1": f1, "AUROC": auroc}
        else:
            return acc, prec, rec, f1


class SegmentationMetrics(Metrics):
    def __init__(self, device, mode="val", task="multiclass"):
        super().__init__(device, mode)
        if task == "multiclass":
            self.iou = torchmetrics.classification.JaccardIndex(task=task, num_classes=3, ignore_index=0, average="macro").to(device)
            self.dice = DiceScore(num_classes=3, include_background=False, input_format="index", average="macro").to(device)
        elif task == "binary":
            self.iou = torchmetrics.classification.JaccardIndex(task=task).to(device)
            self.dice = DiceScore(num_classes=1).to(device)

    def batch_metrics(self, preds, labels):
        preds = torch.argmax(preds, dim=0).unsqueeze(dim=0)
        labels = labels
        self.iou.update(preds, labels)
        self.dice.update(preds, labels)

    def __reset(self):
        self.iou.reset()
        self.dice.reset()

    def epoch_metrics(self):
        iou = self.iou.compute().item()
        dice = self.dice.compute().item()
        self.__reset()
        return {"iou":iou, "dice":dice}


def log_metrics(train_loss, val_loss, acc, prec, rec, f1):
    print(30*"-")
    print(f"Train Loss: {train_loss} | Validation Loss: {val_loss}")
    print(f"Accuracy: {acc} | Precision: {prec} | Recall: {rec} | F1: {f1}")
    print(30 * "-")


def log_wandb(epoch, train_loss, val_loss, acc, prec, rec, f1):
    wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "accuracy": acc, "precision": prec, "recall": rec, "f1": f1})


def log_metrics_segmentation(train_loss, val_loss, iou, dice):
    print(30*"-")
    print(f"Train Loss: {train_loss} | Validation Loss: {val_loss}")
    print(f"IoU: {iou} | Dice: {dice}")
    print(30 * "-")


def log_wandb_segmentation(epoch, train_loss, val_loss, iou, dice):
    wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "IoU": iou, "Dice": dice})

def log_metrics_two_step(train_loss_cls, val_loss_cls, acc, prec, rec, f1, train_loss_seg,
                         val_loss_seg, iou, dice, total_train_loss, total_val_loss):
    print(30*"-")
    print(f"Train Loss: {total_train_loss} | Validation Loss: {total_val_loss}")
    print(f"Train Loss Classification: {train_loss_cls} | Validation Loss Classification: {val_loss_cls}")
    print(f"Train Loss Segmentation: {train_loss_seg} | Validation Loss Segmentation: {val_loss_seg}")
    print(f"Accuracy: {acc} | Precision: {prec} | Recall: {rec} | F1: {f1}")
    print(f"IoU: {iou} | Dice: {dice}")
    print(30 * "-")

def log_wandb_two_step(epoch, train_loss_cls, val_loss_cls, acc, prec, rec, f1, train_loss_seg,
                         val_loss_seg, iou, dice, total_train_loss, total_val_loss):
    wandb.log({"epoch": epoch, "train_loss_total": total_train_loss, "val_loss_total": total_val_loss,
               "train_loss_cls": train_loss_cls, "val_loss_cls": val_loss_cls,
               "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
               "train_loss_seg": train_loss_seg, "val_loss_seg": val_loss_seg,
               "IoU": iou, "Dice": dice})