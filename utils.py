# Source: https://albumentations.ai/docs/examples/tensorflow-example/
# https://github.com/albumentations-team/albumentations/issues/911
# https://www.tensorflow.org/tutorials/load_data/images
# https://www.tensorflow.org/tutorials/load_data/video
# access: 16.06.2023 15:00

import albumentations as alb
import tensorflow as tf
from functools import partial
import cv2
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.scores import BinaryScore
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
import os, re
import pandas as pd


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
 
def get_base_transforms(img_size):
    """
    augment for val and test data
    """ 
    return alb.Compose([
        alb.Resize(height=img_size, width=img_size),
        alb.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT),
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
        tf.data.Dataset
    """
    
    augmentation_func = partial(augment_image, augmentation_pipeline=augmentation_pipeline)
    
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
    augmented_ds = dataset.map(augmentation_func, num_parallel_calls=AUTOTUNE)
    
    return augmented_ds.prefetch(AUTOTUNE)

def get_test_dataset(img_size, batch_size, shuffle):
    df_test_path = os.path.join("data", "test")
    celeb_test_path = os.path.join("data", "test_celeb")
    dfdc_test_path = os.path.join("data", "test_dfdc")
    ff_test_path = os.path.join("data", "test_ff")

    test_data_paths = [df_test_path, celeb_test_path, dfdc_test_path, ff_test_path]
    datasets_test = []
    
    for path in test_data_paths:
        ds = tf.keras.utils.image_dataset_from_directory(
                path,
                labels="inferred",
                label_mode="binary", # 0: fake, 1:real
                image_size=(img_size, img_size),
                batch_size=batch_size,
                shuffle = shuffle
            )
        datasets_test.append(ds)
        
    return datasets_test

def filter_imgs(ds, img_cnt=30):
    img_label_0 = []
    img_label_1 = []
    
    for bilder, labels in ds.as_numpy_iterator():
        for bild, label in zip(bilder, labels):
            if label == 0 and len(img_label_0) < img_cnt:
                img_label_0.append(bild)
            elif label == 1 and len(img_label_1) < img_cnt:
                img_label_1.append(bild)

        if len(img_label_0) >= img_cnt and len(img_label_1) >= img_cnt:
            break
    
    return np.array(img_label_0), np.array(img_label_1)

def plot_model(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


def plot_fine(history_fine, acc, val_acc, loss, val_loss, initial_epochs):
  acc += history_fine.history['accuracy']
  val_acc += history_fine.history['val_accuracy']

  loss += history_fine.history['loss']
  val_loss += history_fine.history['val_loss']

  plt.figure(figsize=(8, 8))
  plt.subplot(2, 1, 1)
  plt.plot(acc, label='Training Accuracy')
  plt.plot(val_acc, label='Validation Accuracy')
  plt.ylim([0, 1.0])
  plt.plot([initial_epochs-1,initial_epochs-1],
            plt.ylim(), label='Start Fine Tuning')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(2, 1, 2)
  plt.plot(loss, label='Training Loss')
  plt.plot(val_loss, label='Validation Loss')
  plt.ylim([0, 1.0])
  plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.xlabel('epoch')
  plt.show()
  
def plot_overview(ds, values, ds_names, value_type):
    # Plotten der Loss-Werte
    plt.figure(figsize=(10, 5))
    bars = plt.bar(range(len(ds)), values)
    plt.xticks(range(len(ds)), ds_names)
    plt.xlabel(f'Testdatensatz')
    plt.ylabel(value_type)
    plt.title(value_type+ ' für verschiedene Testdatensätze')
    
    colors = ['red', 'green', 'blue', 'orange']
    for bar, color in zip(bars, colors):
        bar.set_color(color)
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.4f}', ha='center', va='bottom')
    
    # Legende hinzufügen
    plt.show()
    
def plot_batch_pred_vs_real(datasets, dataset_names, model):
    # Retrieve a batch of images from the test set
    for dataset, name in zip(datasets, dataset_names):
        image_batch, label_batch = dataset.as_numpy_iterator().next()
        predictions = model.predict_on_batch(image_batch).flatten()
        class_names = ["fake", "real"]

        # Apply a sigmoid since our model returns logits
        predictions = tf.nn.sigmoid(predictions)
        predictions = tf.where(predictions < 0.5, 0, 1)
        labels = [int(lb) for lb in label_batch.flatten()]

        print('Predictions:\n', predictions.numpy())
        print('Labels:\n', label_batch.flatten())

        plt.figure(figsize=(10, 10))
        for i in range(10):
            _ = plt.subplot(2, 5, i + 1)
            plt.imshow(image_batch[i].astype("uint8"))
            plt.title(f"Predictions: {class_names[predictions[i]]}\n Actual: {class_names[labels[i]]}")
            plt.axis("off")
            
        plt.suptitle(name, fontsize=16)
        
    plt.tight_layout()     
    plt.show()


def plot_grad_cam(model, label, img):
    score = BinaryScore(0.0)
    replace2linear = ReplaceToLinear()

    # Create Gradcam object
    gradcam = GradcamPlusPlus(model,
                            model_modifier=replace2linear,
                            clone=True)

    # Generate heatmap with GradCAM++
    cam = gradcam(score,
                img,
                penultimate_layer=-1)
    
    plt.figure(figsize=(10, 10))
    for i in range(min(30, len(img))):
        _ = plt.subplot(6, 5, i + 1)
        heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
        plt.imshow(img[i].astype("uint8"))
        plt.imshow(heatmap, cmap='jet', alpha=0.5)
        plt.title(f"Label: {label}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def pred_ds(ds, model):
    # Liste für die Bilder, Labels und Vorhersagen erstellen
    labels = []
    predictions = []
    video_names = []

    for batch_images, batch_labels in ds.as_numpy_iterator():
        # Das Modell auf die Bilder predicten und die Sigmoid-Funktion anwenden
        batch_predictions = model.predict(batch_images, verbose=0)
        batch_predictions = tf.nn.sigmoid(batch_predictions)
        batch_predictions = tf.where(batch_predictions < 0.5, 0, 1)

        # Labels und Vorhersagen in die jeweiligen Listen hinzufügen
        labels.extend(batch_labels)
        predictions.extend(batch_predictions)

    # Die Listen in numpy-Arrays umwandeln
    labels = np.array(labels)
    predictions = np.array(predictions)
    
    for filename in ds.file_paths:
        videoname = re.sub(r'(?:_\d+)?\.jpg$', '.jpg', os.path.basename(filename))
        video_names.append(videoname)
    
    return labels, predictions, video_names

def calculate_video_final_prediction(group, threshhold):
    unique_labels, counts = np.unique(group, return_counts=True)
    percentage = counts / len(group)
    if len(unique_labels) == 2:
        if percentage[0] >= threshhold:
            final_prediction = 0
        else:
            final_prediction = 1
    else:
        final_prediction = unique_labels[percentage >= threshhold][0]
    return final_prediction

def calculate_video_pred(video_ids, labels, predictions, threshhold):
    labels = [int(i) for i in labels.flatten()]
    predictions = predictions.flatten()
    data = {'ID': video_ids, 'Label': labels, 'Prediction': predictions}
    df = pd.DataFrame(data)

    # Gruppieren nach ID und die eindeutigen Labels auswählen
    grouped_df = df.groupby('ID')['Label'].first().reset_index()
    # Endgültige Vorhersage erstellen
    final_predictions = df.groupby('ID')['Prediction'].apply(calculate_video_final_prediction, threshhold=threshhold)
    grouped_df['Final_Prediction'] = final_predictions.reset_index(drop=True)

    return grouped_df

def calculate_video_accuracy(grouped_df):
    correct_predictions = grouped_df[grouped_df['Label'] == grouped_df['Final_Prediction']]
    accuracy = len(correct_predictions) / len(grouped_df) * 100
    return accuracy

def mismatch_videos(df):
    mismatched_ids = df[df['Label'] != df['Final_Prediction']]
    return mismatched_ids