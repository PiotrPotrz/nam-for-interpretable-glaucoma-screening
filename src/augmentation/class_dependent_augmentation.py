import torch
import random
import numpy as np
from skimage.transform import resize
import skimage
import albumentations as A


class InjectorAugmentation(torch.nn.Module):
    def __init__(self, glaucoma_paths, injection_proba=0.3, img_size=(512, 512), crop_border=True):
        super(InjectorAugmentation, self).__init__()
        self.injection_proba = injection_proba
        self.glaucoma_paths = glaucoma_paths
        self.img_size = img_size
        self.basic_augmentation = A.Compose([
            A.Rotate(limit=(-15, 15), p=1.0, crop_border=crop_border),
            A.Resize(512, 512),
        ])

    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    def __read(self, path):
        image = skimage.io.imread(path)
        image = resize(image, self.img_size,
                       anti_aliasing=True, preserve_range=True).astype(np.uint8)
        image = self.basic_augmentation(image=image)["image"]
        return image

    def __replace(self):
        path = random.choice(self.glaucoma_paths)
        image = self.__read(path)
        label  = 1
        return image, label

    def forward(self, img, label):
        if label == 0 and random.random() <= self.injection_proba:
            img, label = self.__replace()
        return img, label

