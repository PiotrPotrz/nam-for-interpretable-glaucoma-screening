from tqdm import tqdm
import torch
from src.utils.metrics import Metrics, SegmentationMetrics
from src.utils.visualization import plot_sample


def val(val_loader, model, loss_fn, device):
    metrics = Metrics(device=device)
    model.eval()
    val_loss = 0
    loop = tqdm(val_loader, desc="   Validation", leave=False)

    with torch.no_grad():
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            loss = loss_fn(outputs.squeeze(0), labels.float())

            val_loss += loss.item()

            metrics.batch_metrics(outputs.squeeze(0), labels)

        acc, prec, rec, f1 = metrics.epoch_metrics()
    return val_loss / len(val_loader), acc, prec, rec, f1


def val_segmentation(val_loader, model, loss_fn, device):
    metrics = SegmentationMetrics(device=device)
    model.eval()
    val_loss = 0
    loop = tqdm(val_loader, desc="   Validation", leave=False)

    with torch.no_grad():
        for i, data in enumerate(loop):
            if len(data) == 3:
                images, labels, _ = data
            else:
                images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            if outputs.shape[1] > 1:
                loss = loss_fn(outputs, labels)
            else:
                loss = loss_fn(outputs.squeeze(0), labels.float())

            val_loss += loss.item()

            metrics.batch_metrics(outputs.squeeze(0), labels)

            if i==0:
                plot_sample(images[0], labels[0], outputs[0])

        metrics = metrics.epoch_metrics()
    return val_loss / len(val_loader), metrics["iou"], metrics["dice"]


def val_two_step(val_loader, model, loss_seg, loss_cls, device):
    seg_metrics = SegmentationMetrics(device=device)
    cls_metrics = Metrics(device=device)

    model.eval()
    val_loss = 0
    val_loss_cls = 0
    val_loss_seg = 0
    loop = tqdm(val_loader, desc="   Validation", leave=False)

    with torch.no_grad():
        for i, data in enumerate(loop):
            images, mask, labels = data

            images, mask, labels = images.to(device), mask.to(device), labels.to(device)

            outputs_cls, outputs_seg = model(images)

            loss_segmentation = loss_seg(outputs_seg.squeeze(1), mask)

            if outputs_cls.shape[1] > 1:
                loss_classification = loss_cls(outputs_cls, labels)
            else:
                loss_classification = loss_cls(outputs_cls.squeeze(0), labels.float())

            loss = loss_segmentation * 2 + loss_classification

            val_loss += loss.item()
            val_loss_cls += loss_classification.item()
            val_loss_seg += loss_segmentation.item()

            seg_metrics.batch_metrics(outputs_seg.squeeze(0), mask)
            cls_metrics.batch_metrics(outputs_cls.squeeze(0), labels)

            if i==0:
                plot_sample(images[0], mask[0], outputs_seg[0])

        seg_metrics = seg_metrics.epoch_metrics()
        acc, prec, rec, f1 = cls_metrics.epoch_metrics()
    return (val_loss / len(val_loader), val_loss_cls/len(val_loader), val_loss_seg/len(val_loader),
            acc, prec, rec, f1, seg_metrics["iou"], seg_metrics["dice"])