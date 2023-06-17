# Code from: https://github.com/erprogs/CViT/tree/main, https://github.com/selimsef/dfdc_deepfake_challenge access: 16.06.2023

import os
import json
import numpy as np
import face_recognition
import cv2
from time import perf_counter
from blazeface import BlazeFace
import torch
import random
import argparse
from tqdm import tqdm
from PIL import Image


def create_folders(dir_path):
    for folder in ["train", "val", "test", "test_dfdc", "test_celeb", "test_ff"]:
        path = os.path.join(dir_path, folder)
        if not os.path.exists(path):
            os.makedirs(path)
        
# load DFDC json
def load_metadata(dir_path):
    metafile = os.path.join(dir_path, "metadata.json")
     
    if os.path.isfile(metafile):
        with open(metafile) as data_file:
            data = json.load(data_file)
    else:
        return None
    
    return data

#dfdc dataset 
def filter_dfdc_files(metadata, num_files):
    real_ids = []
    fake_ids = []

    for video_id, item in metadata.items():
        label = item.get('label')
        if label == 'REAL':
            real_ids.append(video_id)
        elif label == 'FAKE':
            fake_ids.append(video_id)

    random_real_id = random.sample(real_ids, k=num_files)
    random_fake_id = random.sample(fake_ids, k=num_files)
    return random_fake_id+random_real_id


def preprocess_dfdc_files(dir_path, num_frames=1, img_size=(224, 224)):
    # iterate over DFDC dataset
    if os.path.isdir(dir_path):
        destination_path = os.path.join("data", "test_dfdc", str(img_size[0]))
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)
            
        data = load_metadata(dir_path)
        if data:
            filtered_files = filter_dfdc_files(data, 2)
            with tqdm(len(filtered_files)) as bar:
                for filename in filtered_files:
                    # check if the file name is found in metadata, and its label
                    file_path = os.path.join(dir_path, filename)
                    if filename.endswith(".mp4") and os.path.isfile(file_path):
                        label = data[filename]['label'].lower()
                        image_path = os.path.join(destination_path, label)
                        if not os.path.exists(image_path):
                            os.makedirs(image_path)
                        process_video(file_path, filename, image_path, num_frames, img_size)
                    bar.update()
    

def preprocess_celeb_files(dir_path, num_frames=1, img_size=(224, 224)):
     if os.path.isdir(dir_path):
        destination_path = os.path.join("data", "test_celeb", str(img_size[0]))
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)
        
        

# access video
def process_video(video_path, filename, image_path, num_frames, img_size):
    gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    facedet = BlazeFace().to(gpu)
    facedet.load_weights(os.path.join("preprocessing", "blazeface.pth"))
    facedet.load_anchors(os.path.join("preprocessing", "anchors.npy"))
    _ = facedet.train(False)
    
    from video_reader import VideoReader
    from face_extractor import FaceExtractor
     
    video_reader = VideoReader()
    video_read_fn = lambda x: video_reader.read_random_frames(x, num_frames=num_frames)
    face_extractor = FaceExtractor(video_read_fn, facedet)
    
    faces = face_extractor.process_video(video_path) #get list of faces-bbox as numpy arrays with values 0-255 RGB
    # Only look at one face per frame.
    face_extractor.keep_only_best_face(faces)
    
    n = 0
    for frame_data in faces:
        for face in frame_data["faces"]:
            face_locations = face_recognition.face_locations(face) #the detected bbox are further cropped
            img = Image.fromarray(face)
            img.show()
            for face_location in face_locations:
                top, right, bottom, left = face_location
                face_image = face[top:bottom, left:right]
                img = Image.fromarray(face)
                img.show()
                resized_face = cv2.resize(face_image, img_size, interpolation=cv2.INTER_AREA)
                resized_face = cv2.cvtColor(resized_face, cv2.COLOR_RGB2BGR)
                
                destination_path = os.path.join(image_path, filename[:-4])
                cv2.imwrite(destination_path+"_"+str(n)+".jpg", resized_face, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                
                n += 1
                
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="DFDC", type=str, help='Dataset (DFDC / FACEFORENSICS/ Celeb-DF / DF)')
    parser.add_argument('--raw_data_path', default='', type=str, help='Raw Videos directory')
    opt = parser.parse_args()
    
    if opt.dataset.upper() == "DFDC":
        print("Process DFDC data...")
        start_time = perf_counter()
        preprocess_dfdc_files(opt.raw_data_path)
        end_time = perf_counter()
        print("--- %s seconds ---" % (end_time - start_time))

if __name__ == "__main__":
    create_folders("data")
    main()