import torch
import matplotlib
matplotlib.use("Agg")
import os
from torch.utils.data import DataLoader
import wandb
import torch.optim as optim
from lion_pytorch import Lion
from adabelief_pytorch import AdaBelief
from torch.optim.lr_scheduler import CosineAnnealingLR, PolynomialLR, CosineAnnealingWarmRestarts
import segmentation_models_pytorch as smp

from src.train import train_segmentation
from src.val import val_segmentation
from src.dataset import RimOneDataset
from src.utils.metrics import log_metrics, log_wandb, log_metrics_segmentation, log_wandb_segmentation
from src.utils.callbacks import Callback
from src.utils.parse_args import parse_args

from src.augmentation.basic_aug import *

if __name__ == '__main__':
    args = parse_args()

    if "RIM_ONE" in args.dataset:
        classes = 3
    elif "REFUGE" in args.dataset:
        classes = 3
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


    device = torch.device(args.cuda)
    model = smp.create_model(arch=args.segmentor, encoder_name=args.encoder, encoder_weights="imagenet", in_channels=3, classes=classes)
    model.to(device)

    loss_dict = {
        "lovasz": smp.losses.LovaszLoss(mode="multiclass"),
        "dice": smp.losses.DiceLoss(mode="multiclass"),
        "jaccard": smp.losses.JaccardLoss(mode="multiclass"),
        "ce": torch.nn.CrossEntropyLoss(weight=torch.tensor(args.seg_weights).to(device)),
        "tversky": smp.losses.TverskyLoss(mode="multiclass"),
    }


    loss_fn = loss_dict[args.loss]

    epochs = args.epochs
    learning_rate = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    optimizer_dict = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "rmsprop": optim.RMSprop,
        "sgd": optim.SGD,
        "adabelief": AdaBelief,
        "lion": Lion
    }

    if args.optimizer in optimizer_dict:
        if args.optimizer == "sgd":
            optimizer = optimizer_dict[args.optimizer](model.parameters(), lr=args.lr, momentum=args.momentum,
                                                       weight_decay=args.w_decay)
        elif args.optimizer == "adabelief":
            optimizer = optimizer_dict[args.optimizer](model.parameters(), lr=args.lr, weight_decay=args.w_decay,
                                                       eps=1e-16, betas=(0.9, 0.999), weight_decouple=True,
                                                       rectify=False)
        else:
            optimizer = optimizer_dict[args.optimizer](model.parameters(), lr=args.lr, weight_decay=args.w_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    scheduler_dict = {
        "cosine": CosineAnnealingLR(optimizer, T_max=args.epochs),
        "polylr": PolynomialLR(optimizer, total_iters=args.epochs, power=args.power),
        "cosine_wr": CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=args.t0, T_mult=args.tm, eta_min=args.eta)
    }
    scheduler = scheduler_dict[args.scheduler]

    if args.augmentation == "a1":
        aug = augmentation
    elif args.augmentation == "a2":
        aug = augmentation2
    elif args.augmentation == "a3":
        aug = augmentation3
    elif args.augmentation == "a4":
        aug = augmentation4
    else:
        aug = None
    train_dataset = RimOneDataset('train', random_state=7, transformations=aug, dataset=args.dataset,
                                  inject=args.inject, inject_ratio=args.inject_ratio, data_root_dir=args.root, include_segmentation=True)
    test_dataset = RimOneDataset('val', random_state=7, dataset=args.dataset, data_root_dir=args.root, include_segmentation=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, prefetch_factor=args.pf_factor)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers, prefetch_factor=args.pf_factor)


    model_save_name =  (f"SEG{args.model_save_name}${args.segmentor}${args.encoder}${args.loss}$"
                     f"{args.optimizer}${args.epochs}${args.augmentation}${args.scheduler}$"
                     f"{args.lr}${args.pretrained}${args.batch_size}${args.dataset}$")
    model_save_dir = args.models_dir
    if not "tscratch" in model_save_dir:
        model_save_name = f"./{model_save_dir}/{model_save_name}"
        os.makedirs(f"./{model_save_dir}", exist_ok=True)
    else:
        model_save_name = f"{model_save_dir}/{model_save_name}"
        os.makedirs(f"{model_save_dir}", exist_ok=True)

    best_val_loss = float("inf")
    callbacks = Callback(patience=args.patience, model_save_name=model_save_name, mode="seg")

    if args.no_wandb:
        wandb_mode="disabled"
    else:
        wandb_mode = "online"
    wandb.init(project="GLAUCOMA",
               name=model_save_name,
               mode=wandb_mode)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1} || {epochs}")
        tl = train_segmentation(model=model, loss_fn=loss_fn, optimizer=optimizer, device=device, train_loader=train_loader)
        vl,  iou, dice = val_segmentation(model=model, loss_fn=loss_fn, device=device, val_loader=test_loader)
        log_metrics_segmentation(tl, vl, iou, dice)
        log_wandb_segmentation(epoch, tl, vl, iou, dice)
        scheduler.step()

        stop_training = callbacks.on_epoch_end_segmentation(model, epoch, vl, iou, dice)
        if stop_training:
            print("Early stopping triggered")
            break
    callbacks.on_train_end(model)
