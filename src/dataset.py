import torch
import skimage
import numpy as np
import pandas as pd
import os
import warnings

from torch.utils.data import Dataset, DataLoader
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.transform import resize
from sklearn.model_selection import train_test_split

from src.augmentation.class_dependent_augmentation import InjectorAugmentation


class RimOneDataset(Dataset):
    def __init__(self, mode, dataset="RIM_ONE_DL", normalization="minmax", transformations=None,
                 random_state=42, val_size=0.1, inject=False, inject_ratio=0.1, image_size=512, data_root_dir=".",
                 include_segmentation=False):
        self.dataset_name = dataset
        self.segmentation = include_segmentation
        self.mode = mode
        self.transformations = transformations
        self.val_size = val_size
        self.random_state = random_state
        self.inject = inject
        self.inject_ratio = inject_ratio
        self.image_size = image_size
        self.data_root_dir = data_root_dir
        if self.dataset_name == "RIM_ONE_DL":
            self.train_dataset_path = f"{self.data_root_dir}/data/RIM-ONE_DL_images/partitioned_by_hospital/training_set"
            self.test_dataset_path = f"{self.data_root_dir}/data/RIM-ONE_DL_images/partitioned_by_hospital/test_set"
            self.val_dataset_path = ""
        elif self.dataset_name == "RIM_ONE_DL_RANDOM":
            self.train_dataset_path = f"{self.data_root_dir}/data/RIM-ONE_DL_images/partitioned_randomly/training_set"
            self.test_dataset_path = f"{self.data_root_dir}/data/RIM-ONE_DL_images/partitioned_randomly/test_set"
            self.val_dataset_path = ""
        elif self.dataset_name == "EYEPAC_A_Lv2":
            self.train_dataset_path = f"{self.data_root_dir}/data/eyepac-light-v2-512-jpg/train"
            self.test_dataset_path = f"{self.data_root_dir}/data/eyepac-light-v2-512-jpg/test"
            self.val_dataset_path = f"{self.data_root_dir}/data/eyepac-light-v2-512-jpg/validation"
        elif self.dataset_name == "REFUGE":
            self.train_dataset_path = f"{self.data_root_dir}/data/Refuge/REFUGE-Training400/Training400"
            self.val_dataset_path = f"{self.data_root_dir}/data/Refuge/REFUGE-Validation400/REFUGE-Validation400"
            self.test_dataset_path = f"{self.data_root_dir}/data/Refuge/Test400"

            train_masks_dir = f"{self.data_root_dir}/data/Refuge/Annotation-Training400/Disc_Cup_Masks"
            val_masks_dir = f"{self.data_root_dir}/data/Refuge/REFUGE-Validation400-GT/REFUGE-Validation400-GT/Disc_Cup_Masks"
            test_masks_dir = f"{self.data_root_dir}/data/Refuge/REFUGE-Test-GT/Disc_Cup_Masks"
        else:
            raise Exception(f"Wrong dataset name: {self.dataset_name}")

        if normalization == "minmax":
            self.base_transformations = [
                A.Normalize(normalization='min_max'),
                ToTensorV2(),
            ]
        else:
            raise NotImplementedError

        if "RIM_ONE_DL" in self.dataset_name:
            if self.mode == "train" or self.mode == "val":
                path = self.train_dataset_path
            elif self.mode == "test":
                path = self.test_dataset_path
                assert self.transformations is None
            else:
                raise Exception("Wrong mode")
        elif self.dataset_name == "EYEPAC_A_Lv2":
            if self.mode == "train":
                path = self.train_dataset_path
            elif self.mode == "test":
                path = self.test_dataset_path
            elif self.mode == "val":
                path = self.val_dataset_path

        if "RIM_ONE_DL" in self.dataset_name:
            self.paths_glaucoma = sorted(glob.glob(path + "/glaucoma/*.png"))
            self.paths_normal = sorted(glob.glob(path + "/normal/*.png"))
        elif self.dataset_name == "EYEPAC_A_Lv2":
            self.paths_glaucoma = sorted(glob.glob(path + "/RG/*.jpg"))
            self.paths_normal = sorted(glob.glob(path + "/NRG/*.jpg"))
        elif self.dataset_name == "REFUGE":
            if self.mode == "train":
                self.paths_glaucoma = sorted(glob.glob(self.train_dataset_path + "/Glaucoma/*"))
                self.paths_normal = sorted(glob.glob(self.train_dataset_path + "/Non-Glaucoma/*"))
                if self.segmentation:
                    mask_paths_g = sorted(glob.glob(f"{train_masks_dir}/Glaucoma/*"))
                    mask_paths_n = sorted(glob.glob(f"{train_masks_dir}/Non-Glaucoma/*"))
                    self.mask_paths = mask_paths_g + mask_paths_n
            elif self.mode == "val":
                val_df = pd.read_excel(
                    f"{self.data_root_dir}/data/Refuge/REFUGE-Validation400-GT/REFUGE-Validation400-GT/Fovea_locations.xlsx")
                self.paths_glaucoma = sorted(val_df.loc[val_df['Glaucoma Label'] == 1, 'ImgName'].to_list())
                self.paths_normal = sorted(val_df.loc[val_df['Glaucoma Label'] == 0, 'ImgName'].to_list())
                self.paths_glaucoma = [f"{self.val_dataset_path}/" + path for path in self.paths_glaucoma]
                self.paths_normal = [f"{self.val_dataset_path}/" + path for path in self.paths_normal]
                if self.segmentation:
                    # self.mask_paths = sorted(glob.glob(
                    #     f"{self.data_root_dir}/data/Refuge/REFUGE-Validation400-GT/REFUGE-Validation400-GT/Disc_Cup_Masks/*"))
                    self.image_paths = self.paths_glaucoma + self.paths_normal
                    self.mask_paths = [
                        os.path.join(val_masks_dir, os.path.splitext(os.path.basename(img))[0] + ".bmp")
                        for img in self.image_paths
                    ]
                    # print(self.mask_paths)
            elif self.mode == "test":
                test_df = pd.read_excel(
                    f"{self.data_root_dir}/data/Refuge/REFUGE-Test-GT/Glaucoma_label_and_Fovea_location.xlsx")
                self.paths_glaucoma = sorted(test_df.loc[test_df['Label(Glaucoma=1)'] == 1, 'ImgName'].to_list())
                self.paths_normal = sorted(test_df.loc[test_df['Label(Glaucoma=1)'] == 0, 'ImgName'].to_list())
                self.paths_glaucoma = [f"{self.test_dataset_path}/" + path for path in self.paths_glaucoma]
                self.paths_normal = [f"{self.test_dataset_path}/" + path for path in self.paths_normal]
                if self.segmentation:
                    mask_paths_g = sorted(glob.glob(f"{test_masks_dir}/G/*"))
                    mask_paths_n = sorted(glob.glob(f"{test_masks_dir}/N/*"))
                    self.mask_paths = mask_paths_g + mask_paths_n
        self.image_paths = self.paths_glaucoma + self.paths_normal
        self.labels = [1] * len(self.paths_glaucoma) + [0] * len(self.paths_normal)

        if "RIM_ONE_DL" in self.dataset_name:
            if self.mode == "train" or self.mode == "val":
                X_train, X_val, y_train, y_val = train_test_split(self.image_paths, self.labels,
                                                                  test_size=val_size, random_state=random_state,
                                                                  stratify=self.labels)
                if self.mode == "train":
                    self.image_paths = X_train
                    self.labels = y_train
                elif self.mode == "val":
                    self.image_paths = X_val
                    self.labels = y_val
        if self.inject and self.mode == "train":
            if "RIM_ONE" in self.dataset_name:
                self.paths_glaucoma = [self.image_paths[i] for i in range(len(self.image_paths)) if self.labels[i] == 1]
                crop_border = True
            else:
                crop_border = False

            self.injector = InjectorAugmentation(glaucoma_paths=self.paths_glaucoma,
                                                 injection_proba=self.inject_ratio,
                                                 img_size=(self.image_size, self.image_size), crop_border=crop_border)

        #
        if self.segmentation:
            if self.inject:
                warnings.warn("Warning: Injection mode not supported for segmentation yet!")
            if "RIM_ONE" in self.dataset_name:
                self.unified_paths = self.__process_paths_rim_one_dl()
            elif "REFUGE" in self.dataset_name:
                pass
            else:
                raise Exception("Segmentation mode not supported for dataset ", self.dataset_name)

    def __len__(self):
        return len(self.image_paths)

    def __process_paths_rim_one_dl(self):
        ann = glob.glob(f"{self.data_root_dir}/data/RIM-ONE_DL_reference_segmentations/glaucoma/*.png") + glob.glob(
            f"{self.data_root_dir}/data/RIM-ONE_DL_reference_segmentations/normal/*.png")

        ann_cup = sorted([path for path in ann if "Cup" in path])
        ann_disc = sorted([path for path in ann if "Disc" in path])

        df = pd.DataFrame({"cup": ann_cup, "disc": ann_disc})
        df["base_name"] = df["cup"].apply(lambda x: x.split(os.sep)[-1].replace("-1-Cup-T.png", ""))

        df_paths = pd.DataFrame({"images": self.image_paths, "labels": self.labels})
        df_paths["base_name"] = df_paths["images"].apply(lambda x: x.split(os.sep)[-1].replace(".png", ""))

        merged = pd.merge(df_paths, df, on='base_name', how='inner')

        return merged

    def __getitem__(self, idx):
        if self.transformations is not None:
            transform = A.Compose(self.transformations + self.base_transformations)
        else:
            transform = A.Compose(self.base_transformations)

        if self.segmentation:
            if "RIM_ONE_DL" in self.dataset_name:
                unified_paths = self.unified_paths.loc[idx]
                img, cup, disc, label = unified_paths[["images", "cup", "disc", "labels"]]

                image = skimage.io.imread(img)
                image = resize(image, (512, 512),
                               anti_aliasing=True, preserve_range=True).astype(np.uint8)

                cup_mask = skimage.io.imread(cup)
                cup_mask = resize(cup_mask, (512, 512),
                                  preserve_range=True)
                cup_mask = (cup_mask > 128).astype(np.uint8)

                disc_mask = skimage.io.imread(disc)
                disc_mask = resize(disc_mask, (512, 512),
                                   preserve_range=True)
                disc_mask = (disc_mask > 128).astype(np.uint8)

                mask = disc_mask + cup_mask
                transformed = transform(image=image, mask=mask)

                image = transformed["image"]
                mask = transformed["mask"]

                label = torch.tensor(label, dtype=torch.long)
                mask = mask.long()
            elif self.dataset_name == "REFUGE":
                image_path = self.image_paths[idx]
                label = self.labels[idx]

                image = skimage.io.imread(image_path)

                image = resize(image, (self.image_size, self.image_size),
                               anti_aliasing=True, preserve_range=True).astype(np.uint8)

                mask = skimage.io.imread(self.mask_paths[idx])

                mask2 = np.zeros_like(mask)
                mask2[mask == 0] = 0
                mask2[mask == 128] = 1
                mask2[mask == 255] = 2
                mask = mask2
                mask = resize(mask, (self.image_size, self.image_size),
                              preserve_range=True)

                transformed = transform(image=image, mask=mask)

                image = transformed["image"]
                mask = transformed["mask"]

                label = torch.tensor(label, dtype=torch.long)
                mask = mask.long()

            return image, mask, label
        else:
            image_path = self.image_paths[idx]
            label = self.labels[idx]

            image = skimage.io.imread(image_path)

            image = resize(image, (self.image_size, self.image_size),
                           anti_aliasing=True, preserve_range=True).astype(np.uint8)

            if self.inject and self.mode == "train":
                image, label = self.injector(image, label)

            image = transform(image=image)["image"]
            label = torch.tensor(label, dtype=torch.long)

            return image, label

if __name__ == "__main__":
    train_dataset = RimOneDataset("train", normalization="minmax", dataset="RIM_ONE_DL_RANDOM", inject=True, inject_ratio=0.5)
    val_dataset = RimOneDataset("val", normalization="minmax", dataset="REFUGE")
    test_dataset = RimOneDataset("test", normalization="minmax", dataset="REFUGE")

    train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    print("Train loader size:", len(train_loader))
    print("Val loader size:", len(val_loader))
    print("Test loader size:", len(test_loader))
    for image, label in train_loader:
        print(f"image shape: {image.shape} || label shape: {label.shape}")
        print(f"image min: {image.min()} || max: {image.max()}")
        print("Label:", label)
        break

    for image, label in val_loader:
        print(f"image shape: {image.shape} || label shape: {label.shape}")
        print(f"image min: {image.min()} || max: {image.max()}")
        print("Label:", label)
        break

    for image, label in test_loader:
        print(f"image shape: {image.shape} || label shape: {label.shape}")
        print(f"image min: {image.min()} || max: {image.max()}")
        print("Label:", label)
        break



    train_dataset = RimOneDataset("train", normalization="minmax", dataset="REFUGE", include_segmentation=True)
    val_dataset = RimOneDataset("val", normalization="minmax", dataset="REFUGE", include_segmentation=True)
    test_dataset = RimOneDataset("test", normalization="minmax", dataset="REFUGE", include_segmentation=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    print("Train loader size:", len(train_loader))
    print("Val loader size:", len(val_loader))
    print("Test loader size:", len(test_loader))
    for image, mask, label in train_loader:
        print(f"image shape: {image.shape} || mask shape: {mask.shape} || label shape: {label.shape}")
        print(f"image min: {image.min()} || max: {image.max()}")
        print("Label:", label)
        break

    for image, mask, label in val_loader:
        print(f"image shape: {image.shape} || mask shape: {mask.shape} || label shape: {label.shape}")
        print(f"image min: {image.min()} || max: {image.max()}")
        print("Label:", label)
        break

    for image, mask, label in test_loader:
        print(f"image shape: {image.shape} || mask shape: {mask.shape} || label shape: {label.shape}")
        print(f"image min: {image.min()} || max: {image.max()}")
        print("Label:", label)
        break


