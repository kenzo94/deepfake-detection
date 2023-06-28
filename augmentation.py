# Source: https://albumentations.ai/docs/examples/tensorflow-example/
# https://github.com/albumentations-team/albumentations/issues/911
# access: 16.06.2023 15:00

import albumentations as alb
import tensorflow as tf
from functools import partial
import cv2
import numpy as np

#color saturation change (CS), color contrast change
#(CC), local block-wise (BW), white Gaussian noise in color
# components (GNC), Gaussian blur (GB) and JPEG compression (JPEG). gewinner der challenge für df

""" „data augmentations: horizontal flip, Gaussian Noise/ISO Noise, blur (motion blur and Gaussian blur), 
random hue-saturation modification, random brightness, contrast modification, image sharpening and embossing, 
and lastly adding a sepia filter.“ ([Silva et al., 2022, p. 6] """

""" „1. Gaussian blur 2. Gaussian noise 3. Motion blur 4. Homomorphic filter enhancement 5. Fourier transform (magnitude spectrum)“
 """
 
"""  „Second, different random augmentations are applied such as rotation, horizontal and vertical flipping and change of the color attributes of the image.“
 ([Khalil et al., 2021, p. 6](zotero://select/library/items/KM9UEHJT)) """
 
"""  „We also applied data augmentation techniques to diversify the training data. We varied the following conditions: 
     1) Brightness (-30% to 30%), 2) Channel shift (-50 to 50), 
 3) Zoom“ ([Tariq et al., 2021, p. 8](zotero://select/library/items/XQKI4CP6)) 
 ([pdf](zotero://open-pdf/library/items/WKGGWL33?page=8)) „(-20% to 20%), 
 4) Rotation (-30◦ degrees to 30◦), and 5) Horizontal flip (50% probability).“ 
 """

def normalize():
    return alb.Compose([
        alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

def get_train_transforms(img_size, p):
    """augment train data

    Args:
        img_size (int): _resize img to
        p (float): probability of applying all list of transforms

    Returns:
        _type_: augmentation pipeline
    """
    return alb.Compose([
        alb.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        alb.GaussNoise(p=0.1),
        alb.GaussianBlur(blur_limit=3, p=0.05),
        alb.HorizontalFlip(),
        alb.VerticalFlip(),
        alb.Resize(height=img_size, width=img_size),
        alb.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT),
        alb.OneOf([alb.RandomBrightnessContrast(), alb.HueSaturationValue(), 
                   alb.Emboss(), alb.Sharpen(), alb.CLAHE(clip_limit=2)], p=0.5),
        alb.OneOf([alb.CoarseDropout(), alb.GridDropout()], p=0.2), #cutout regions in the picture
        alb.OneOf([alb.ToGray(), alb.ToSepia()], p=0.2),
        alb.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5)
    ], p=p
    )
 
def get_base_transforms(img_size, p):
    """
    augment for val and test data
    """ 
    return alb.Compose([
        alb.Resize(height=img_size, width=img_size),
        alb.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT),
        alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

def augment_image(imgs, labels, augmentation_pipeline: alb.Compose):
    def apply_augmentation(images):
        augmented_images = []

        for img in images:  # apply augmentation pipeline to single image
            aug_data = augmentation_pipeline(image=img.astype("uint8"))
            augmented_images.append(aug_data['image'])

        return np.stack(augmented_images)

    inputs = tf.numpy_function(func=apply_augmentation, inp=[imgs], Tout=tf.uint8)

    return inputs, labels


def get_tf_dataset(dataset_path: str, augmentation_pipeline: alb.Compose, batch_size: int = 32, image_size= (224, 224), seed: int = 42, shuffle=True) -> tf.data.Dataset:
    """get augmented dataset in tf dataset format

    Args:
        dataset_path (str): path of dataset
        augmentation_pipeline (alb.Compose): _description_
        batch_size (int, optional): _description_. Defaults to 32.
        image_size (tuple, optional): _description_. Defaults to (224, 224).
        seed (int, optional): _description_. Defaults to 42.
        shuffle (bool, optional): _description_. Defaults to True.

    Returns:
        tf.data.Dataset: _description_
    """
    augmentation_func = partial(augment_image, augmentation_pipeline=augmentation_pipeline,)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        labels="inferred",
        label_mode="binary",
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
        shuffle=shuffle
    )
    return dataset.map(augmentation_func, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)