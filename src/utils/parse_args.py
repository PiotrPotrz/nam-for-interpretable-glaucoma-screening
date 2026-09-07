import argparse

import argparse
import datetime
import ast


def parse_args():
    """
    Funkcja pozwalająca na wprowadzanie argumentów z terminala podczas treningu modeli.
    Returns
    -------
    Argumenty używane w dalszej części kodu.
    """
    now = datetime.datetime.now()
    formatted_now = now.strftime("%Y_%m_%d %H_%M_%S")
    parser = argparse.ArgumentParser(
        description="Training script for glaucoma classification."
    )

    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")
    parser.add_argument(
        "--model_save_name",
        type=str,
        default=formatted_now,
        help="Specify model save name",
    )
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer")
    parser.add_argument("--model", type=str, default="resnet18", help="Specify model name")
    parser.add_argument("--loss", type=str, default="bce", help="Specify loss function")
    parser.add_argument(
        "--augmentation", type=str, default=None, help="Augmentation config"
    )
    parser.add_argument(
        "--cuda", type=str, default="cuda:0", help="Specify cuda device"
    )
    parser.add_argument(
        "--models_dir", type=str, default="saved_models", help="Specify model save directory"
    )
    parser.add_argument(
        "--pretrained", action='store_false', help="Specify pretrained encoder"
    )
    parser.add_argument(
        "--scheduler", type=str, default="cosine", help="Specify scheduler type"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Specify patience"
    )
    parser.add_argument(
        "--workers", type=int, default=2, help="Number of workers used during dataset loading"
    )
    parser.add_argument(
        "--pf_factor", type=int, default=2, help="Number of workers used during dataset loading"
    )
    parser.add_argument(
        "--pin_mem", action='store_false', help="Using pin memory?"
    )
    parser.add_argument(
        "--inject", action='store_true', help="Enables replacing negative samples with positive ones."
    )
    parser.add_argument(
        "--inject_ratio", type=float, default=0.2, help="Injection ratio - how probable is replacing negative sample with positive one."
    )
    parser.add_argument(
        "--w_decay", type=float, default=0.0, help="Specify weight decay?"
    )
    parser.add_argument(
        "--power", type=float, default=0.9, help="Specify the lr decaly power while using POLY LR"
    )
    parser.add_argument(
        "--t0", type=int, default=80, help="Set t0 for warm restarts."
    )
    parser.add_argument(
        "--tm", type=int, default=2, help="Multiplication factor for warm restarts."
    )
    parser.add_argument(
        "--eta", type=float, default=1e-6, help="Eta for warm restarts."
    )
    parser.add_argument(
        "--pos_weight", type=float, default=1.0, help="Weight for positive class loss."
    )
    parser.add_argument(
        "--size", type=int, default=512, help="Size of images processed by network." #TODO
    )
    parser.add_argument("--dataset", type=str, default="RIM_ONE_DL", help="Specify dataset type")
    parser.add_argument("--root", type=str, default=".", help="Specify dataset root directory")
    parser.add_argument(
        "--no_wandb", action='store_true', help="Disables wandb logging."
    )
    parser.add_argument("--segmentor", type=str, default="unet", help="Specifiy segmentor model eg. unet")
    parser.add_argument("--encoder", type=str, default="resnet18", help="Specify encoder name. Remember to use tu- if model is from TIMM.")
    parser.add_argument(
        "--seg_weights", type=ast.literal_eval, default=(1.0, 1.0, 1.0), help="Weights for multichannel loss - custom partial loss. (bg, head, tail)"
    )
    parser.add_argument(
        "--deep_s", type=ast.literal_eval, default=(1.0, 0.5, 0.5, 0.5), help="Deep supervsion weights for different stage losses"
    )
    parser.add_argument(
        "--extractor", type=ast.literal_eval, default=[[3, 8], [8, 16], [16, 32], [32, 1]], help="Deep supervsion weights for different stage losses"
    )
    parser.add_argument(
        "--processor", type=ast.literal_eval, default=[[1, 8], [8, 16], [16, 32], [32, 1]], help="Deep supervsion weights for different stage losses"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed used for training."
    )
    return parser.parse_args()

def parse_args_inference():
    """
    Funkcja pozwalająca na wprowadzanie argumentów z terminala podczas treningu modeli.
    Returns
    -------
    Argumenty używane w dalszej części kodu.
    """
    now = datetime.datetime.now()
    formatted_now = now.strftime("%Y_%m_%d %H_%M_%S")
    parser = argparse.ArgumentParser(
        description="Training script for glaucoma classification."
    )
    parser.add_argument(
        "--cuda", type=str, default="cuda:0", help="Specify cuda device"
    )
    parser.add_argument(
        "--models_dir", type=str, default="saved_models", help="Specify model save directory"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of workers used during dataset loading"
    )
    parser.add_argument(
        "--pf_factor", type=int, default=8, help="Number of workers used during dataset loading"
    )
    parser.add_argument(
        "--pin_mem", action='store_false', help="Using pin memory?"
    )
    parser.add_argument("--root", type=str, default=".", help="Specify dataset root directory")
    parser.add_argument("--results_dir", type=str, default=".", help="Specify results root directory")

    return parser.parse_args()
