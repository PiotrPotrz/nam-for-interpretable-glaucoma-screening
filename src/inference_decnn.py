import torch
import timm
from tqdm import tqdm
import glob
from torch.utils.data import DataLoader
import pandas as pd
import datetime
import os

import src.models.decnn
from src.utils.parse_args import parse_args_inference
from src.dataset import RimOneDataset
from src.utils.metrics import Metrics
from src.utils.utils import  add_result

if __name__=='__main__':
    args = parse_args_inference()
    device = torch.device(args.cuda)

    formatted_now = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M:%S")


    columns = ["SaveName", "Model",
               "Loss", "Optimizer", "Epochs",
               "Augmentation", "Scheduler", "lr",
               "Pretrained", "Batch size",
               "Dataset", "Accuracy", "Precision", "Recall", "F1", "AUROC",
               "type", "path"]

    results_df = pd.DataFrame(columns=columns)

    number_of_models = len(glob.glob(f'./{args.models_dir}/*.pth'))
    print(f"Found {number_of_models} models to perform inference on in {args.models_dir}")
    for i, path in enumerate(sorted(glob.glob(f'{args.models_dir}/*.pth'))):
        name_split = path.split('$')
        print(f"[{i+1}/{number_of_models}] Evaluating model: {path[1]} on dataset {name_split[10]} - checkpoint: {name_split[-1]}")
        model = src.models.decnn.DECNN_WITH_SEGMENTOR().to(device)
        test_dataset = RimOneDataset('test', dataset=name_split[10], data_root_dir=args.root, include_segmentation=True)

        model.load_state_dict(torch.load(path))

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers, prefetch_factor=args.pf_factor)

        metrics = Metrics(device=device, mode="test")
        model.eval()
        loop = tqdm(test_loader, desc="   Testing", leave=False)

        with torch.no_grad():
            for images, masks, labels in loop:
                images, masks, labels = images.to(device), masks.to(device), labels.to(device)

                outputs, _ = model(images)

                metrics.batch_metrics(outputs.squeeze(0), labels)

            results = metrics.epoch_metrics()
        results_df = add_result(results_df, path, results)
        print(f"Accuracy: {results["Accuracy"]} | Precision {results["Precision"]} |  Recall {results["Recall"]} | F1 {results["F1"]} | AUROC {results["AUROC"]}")
    os.makedirs(f'{args.results_dir}/wyniki', exist_ok=True)
    results_df.to_csv(f"{args.results_dir}/wyniki/{args.models_dir.replace(os.sep,"_")}_{formatted_now}_results_inference.csv", index=False)