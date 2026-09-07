import torch
from collections import defaultdict
from torch.utils.data import DataLoader
from src.dataset import RimOneDataset
from tqdm import tqdm

def count_mask_classes(loader, num_classes=None, device="cpu"):
    counts = defaultdict(int)

    for image, mask, label in tqdm(loader):
        mask = mask.to(device)

        # jeśli mask ma wymiar [B, 1, H, W] → [B, H, W]
        if mask.ndim == 4 and mask.shape[1] == 1:
            mask = mask.squeeze(1)

        # flatten wszystkich masek w batchu
        flat = mask.view(-1)

        if num_classes is not None:
            bincount = torch.bincount(flat, minlength=num_classes)
        else:
            bincount = torch.bincount(flat)

        for cls, cnt in enumerate(bincount.tolist()):
            counts[cls] += cnt

    return dict(counts)


if __name__ == "__main__":

    train_dataset = RimOneDataset('train', random_state=7, dataset="RIM_ONE_DL_RANDOM",
                                  include_segmentation=True)
    loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, prefetch_factor=4)



    class_counts = count_mask_classes(loader, num_classes=3)
    print(class_counts)


    def normalize_counts(counts):
        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()}


    freqs = normalize_counts(class_counts)
    print(freqs)