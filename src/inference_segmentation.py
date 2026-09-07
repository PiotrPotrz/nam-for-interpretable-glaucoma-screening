import torch
import segmentation_models_pytorch as smp
from tqdm import tqdm
import glob
from torch.utils.data import DataLoader
import pandas as pd
import datetime
import os

from src.utils.parse_args import parse_args_inference
from src.dataset import RimOneDataset
from src.utils.utils import  add_result
from src.utils.metrics import SegmentationMetrics

if __name__=='__main__':
    args = parse_args_inference()
    device = torch.device(args.cuda)

    formatted_now = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M:%S")


    columns = ["SaveName", "Segmentor", "Encoder",
               "Loss", "Optimizer", "Epochs",
               "Augmentation", "Scheduler", "lr",
               "Pretrained", "Batch size",
               "Dataset", "iou", "dice",
               "type", "path"]

    results_df = pd.DataFrame(columns=columns)

    number_of_models = len(glob.glob(f'./{args.models_dir}/*.pth'))
    print(f"Found {number_of_models} models to perform inference on in {args.models_dir}")
    for i, path in enumerate(sorted(glob.glob(f'{args.models_dir}/*.pth'))):
        name_split = path.split('$')
        print(f"[{i+1}/{number_of_models}] Evaluating model: {name_split[1]} with encoder {name_split[2]} on dataset {name_split[11]} - checkpoint: {name_split[-1]}")

        if "RIM_ONE" in name_split[11]:
            classes = 3
        elif "REFUGE" in name_split[11]:
            classes = 3
        else:
            raise ValueError(f"Unknown dataset: {name_split[11]}")

        model = smp.create_model(arch=name_split[1], encoder_name=name_split[2],
                                 in_channels=3, classes=classes).to(device)
        test_dataset = RimOneDataset('test', dataset=name_split[11], data_root_dir=args.root, include_segmentation=True)

        model.load_state_dict(torch.load(path))

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers, prefetch_factor=args.pf_factor)

        metrics = SegmentationMetrics(device=device, mode="test")

        model.eval()
        loop = tqdm(test_loader, desc="   Testing", leave=False)

        with torch.no_grad():
            for data in loop:
                if len(data) == 3:
                    images, labels, _ = data
                else:
                    images, labels = data
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)

                metrics.batch_metrics(outputs.squeeze(0), labels)

            results = metrics.epoch_metrics()
        results_df = add_result(results_df, path, results)
        print(f"IoU: {results["iou"]} | Dice {results["dice"]}")
    os.makedirs(f'{args.results_dir}/wyniki', exist_ok=True)
    results_df.to_csv(f"{args.results_dir}/wyniki/{args.models_dir.replace(os.sep,"_")}_{formatted_now}_results_inference.csv", index=False)