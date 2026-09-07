import albumentations as A
import cv2

augmentation = [
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=0.1),
    A.Rotate(limit=(-5, 5), p=0.1, crop_border=True),
    A.ElasticTransform(alpha=50.0, sigma=10.0, p=0.1),
    A.ISONoise(p=0.1),
    A.Resize(512, 512, p=1.0),
]

# best for cropped fungus
augmentation2 = [
    A.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.1, hue=0.02, p=0.2),
    A.HorizontalFlip(p=0.2),
    A.Rotate(limit=(-30, 30), p=0.3, crop_border=True),
    A.ElasticTransform(alpha=300.0, sigma=10.0, p=0.2,  border_mode=cv2.BORDER_REFLECT_101),
    A.Defocus(radius=(1, 5), p=0.2),
    A.Affine(
    scale=(0.85, 1.1),
    translate_percent=None,
    rotate=0,
    shear=0,
    p=0.3,
    border_mode=cv2.BORDER_REFLECT_101
    ),
    A.Resize(512, 512, p=1.0),
]

# full fungus
augmentation3 = [
    A.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.1, hue=0.02, p=0.2),
    A.HorizontalFlip(p=0.2),
    A.Rotate(limit=(-30, 30), p=0.3, crop_border=False),
    A.ElasticTransform(alpha=100.0, sigma=10.0, p=0.1,  border_mode=cv2.BORDER_REFLECT_101),
    A.Defocus(radius=(1, 5), p=0.2),
    A.Affine(
    scale=(1.0, 1.05),
    translate_percent=None,
    rotate=0,
    shear=0,
    p=0.3,
    border_mode=cv2.BORDER_REFLECT_101
    ),
    A.Resize(512, 512, p=1.0),
]


# for cropped fungus

augmentation4 = [
    A.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.1, hue=0.02, p=0.25),
    A.HorizontalFlip(p=0.2),
    A.Rotate(limit=(-30, 30), p=0.4, crop_border=True),
    A.ElasticTransform(alpha=300.0, sigma=10.0, p=0.3,  border_mode=cv2.BORDER_REFLECT_101),
    A.Defocus(radius=(1, 5), p=0.2),
    A.Affine(
    scale=(0.8, 1.2),
    translate_percent=None,
    rotate=0,
    shear=0,
    p=0.3,
    border_mode=cv2.BORDER_REFLECT_101
    ),
    A.Resize(512, 512, p=1.0),
]

# best for cropped fungus
delicate_augmentation = [
    A.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.1, hue=0.02, p=0.2),
    A.HorizontalFlip(p=0.3),
    A.Rotate(limit=(-10, 10), p=0.4, crop_border=True),
    A.Affine(
    scale=(0.85, 1.1),
    translate_percent=None,
    rotate=0,
    shear=0,
    p=0.4,
    border_mode=cv2.BORDER_REFLECT_101
    ),
    A.Resize(512, 512, p=1.0),
]