import torch
import timm
from torch.utils.data import DataLoader
import wandb
import torch.optim as optim
from lion_pytorch import Lion
from adabelief_pytorch import AdaBelief
from torch.optim.lr_scheduler import CosineAnnealingLR, PolynomialLR, CosineAnnealingWarmRestarts
import os

from src.train import train
from src.val import val
from src.dataset import RimOneDataset
from src.utils.metrics import log_metrics, log_wandb
from src.utils.callbacks import Callback
from src.utils.parse_args import parse_args

from src.augmentation.basic_aug import *
from src.utils.seed import seed_everything, seed_worker

if __name__ == '__main__':
    args = parse_args()
    seed = args.seed
    seed_everything(seed)

    g = torch.Generator()
    g.manual_seed(seed)

    device = torch.device(args.cuda)
    model = timm.create_model(args.model, pretrained=args.pretrained, num_classes=1)
    model.to(device)

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight]).to(device))

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
                                  inject=args.inject, inject_ratio=args.inject_ratio, data_root_dir=args.root)
    test_dataset = RimOneDataset('val', random_state=7, dataset=args.dataset, data_root_dir=args.root)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers,
                              prefetch_factor=args.pf_factor,
                              worker_init_fn=seed_worker,
                              generator=g
                              )
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=args.workers,
                             prefetch_factor=args.pf_factor,
                             )


    model_save_name =  (f"s_{seed}_{args.model_save_name}${args.model}${args.loss}$"
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
    callbacks = Callback(patience=args.patience, model_save_name=model_save_name)

    if args.no_wandb:
        wandb_mode="disabled"
    else:
        wandb_mode = "online"
    wandb.init(project="GLAUCOMA",
               name=model_save_name, mode=wandb_mode)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1} || {epochs}")
        tl = train(model=model, loss_fn=loss_fn, optimizer=optimizer, device=device, train_loader=train_loader)
        vl,  acc, prec, rec, f1 = val(model=model, loss_fn=loss_fn, device=device, val_loader=test_loader)
        log_metrics(tl, vl, acc, prec, rec, f1)
        log_wandb(epoch, tl, vl, acc, prec, rec, f1)
        scheduler.step()

        stop_training = callbacks.on_epoch_end(model, epoch, vl, f1, rec, prec)
        if stop_training:
            print("Early stopping triggered")
            break
    callbacks.on_train_end(model)
