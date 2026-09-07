import os
import torch
import pandas as pd

from torch.utils.data import DataLoader
from tqdm import tqdm

import src.models.decnn
from src.dataset import RimOneDataset
from src.utils.parse_args import parse_args
from src.utils.seed import seed_everything, seed_worker

from src.augmentation.basic_aug import *


def extract_to_csv(model, loader, device, output_path):
    model.eval()

    columns = model.get_feature_names()
    first_batch = not os.path.exists(output_path)

    with torch.no_grad():
        loop = tqdm(loader, desc=f"Extracting -> {output_path}", leave=False)

        for images, masks, labels in loop:
            images = images.to(device)

            features = model(images)  # [B, 65]

            df = pd.DataFrame(
                features.detach().cpu().numpy(),
                columns=columns
            )

            df["label"] = labels.cpu().numpy()

            df.to_csv(
                output_path,
                mode="a",
                header=first_batch,
                index=False
            )

            first_batch = False


if __name__ == '__main__':
    args = parse_args()

    seed_everything(args.seed)

    g = torch.Generator()
    g.manual_seed(args.seed)

    device = torch.device(args.cuda)

    model = src.models.decnn.FEATURE_EXTRACTOR_WITH_SEGMENTOR(
        dataset=args.dataset
    )
    model.to(device)

    if args.augmentation == "a1":
        aug = augmentation
    elif args.augmentation == "a2":
        aug = augmentation2
    elif args.augmentation == "a3":
        aug = augmentation3
    elif args.augmentation == "a4":
        aug = augmentation4
    elif args.augmentation == "delicate":
        aug = delicate_augmentation
    else:
        aug = None

    train_dataset = RimOneDataset(
        'train',
        random_state=7,
        transformations=aug,
        dataset=args.dataset,
        inject=args.inject,
        inject_ratio=args.inject_ratio,
        data_root_dir=args.root,
        include_segmentation=True
    )

    test_dataset = RimOneDataset('test',
                                 dataset=args.dataset,
                                 data_root_dir=args.root,
                                 include_segmentation=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        prefetch_factor=args.pf_factor,
        worker_init_fn=seed_worker,
        generator=g
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        prefetch_factor=args.pf_factor
    )

    output_dir = args.models_dir
    os.makedirs(output_dir, exist_ok=True)

    train_csv = os.path.join(output_dir, f"s{args.seed}_{args.dataset}_train_logreg_features.csv")
    test_csv = os.path.join(output_dir, f"{args.dataset}_test_logreg_features.csv")

    if os.path.exists(train_csv):
        os.remove(train_csv)

    both = True
    if os.path.exists(test_csv):
        both = False

    extract_to_csv(
        model=model,
        loader=train_loader,
        device=device,
        output_path=train_csv
    )

    if both:
        extract_to_csv(
            model=model,
            loader=test_loader,
            device=device,
            output_path=test_csv
        )

    print(f"Saved train features to: {train_csv}")
    if both:
        print(f"Saved val features to: {test_csv}")