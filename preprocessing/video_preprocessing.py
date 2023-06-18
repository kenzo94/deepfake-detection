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
import re 


        
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

def filter_celeb_files(file_path, num_files):
    random_id = random.sample(file_path, k=num_files)
    return random_id

def filter_df_files(dir_path, list_path):
    fakes = []
    reals = []
    dataset = dict()
    path_raw = []
    
    for dir in os.listdir(dir_path):
        path = os.path.join(dir_path, dir)
        path_raw.append(path)
        path_raw.sort()
        
    with open(list_path) as file:
        while True:
            line = file.readline().replace("\n", "")
            if not line:
                break
            fakes.append(os.path.join(path_raw[0], line))
            real = re.sub(r"_.*\.mp4", ".mp4", line)
            reals.append(os.path.join(path_raw[1], real))
                
    dataset["fakes"] = fakes
    dataset["reals"] = reals
    return dataset

def process_df_videos(dataset, destination, num_frames, img_size):
    with tqdm(len(dataset["fakes"]+dataset["reals"])) as bar:
        fake = dataset["fakes"]
        real = dataset["reals"]
        for video_path in fake:
            image_path = os.path.join(destination, "fake")
            filename = os.path.basename(video_path)
            if not os.path.exists(image_path):
                os.makedirs(image_path)
            process_video(video_path, filename, image_path, num_frames, img_size)
            bar.update()
        for video_path in real:
            image_path = os.path.join(destination, "real")
            filename = os.path.basename(video_path)
            if not os.path.exists(image_path):
                os.makedirs(image_path)
            process_video(video_path, filename, image_path, num_frames, img_size)
            bar.update()
                
                
def preprocess_dfdc_files(dir_path, num_frames=1, img_size=(224, 224)):
    # iterate over DFDC dataset
    if os.path.isdir(dir_path):
        destination_path = os.path.join("data", "test_dfdc")
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
        destination_path = os.path.join("data", "test_celeb")
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)
        
        for dir in os.listdir(dir_path):
            path = os.path.join(dir_path, dir)
            if os.path.isdir(path):
                files = filter_celeb_files(os.listdir(path), 2)
                with tqdm(len(files)) as bar:
                    for file in files:
                        file_path = os.path.join(path, file)
                        if dir[6:] == "real":
                            image_path = os.path.join(destination_path, "real")
                            if not os.path.exists(image_path):
                                os.makedirs(image_path)
                            process_video(file_path, file, image_path, num_frames, img_size)
                        elif dir[6:] == "synthesis":
                            image_path = os.path.join(destination_path, "fake")
                            if not os.path.exists(image_path):
                                os.makedirs(image_path)
                            process_video(file_path, file, image_path, num_frames, img_size)
                        bar.update()
                        
def preprocess_ff_files(dir_path, num_frames=1, img_size=(224, 224)):
    if os.path.isdir(dir_path):
        destination_path = os.path.join("data", "test_ff")
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)
            
        target_folder = "videos"
        manipulated_paths = []
        original_paths = []
        # Rekursiv alle Unterordner durchsuchen
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith(".mp4") and target_folder in root:
                    file_path = os.path.join(root, file)
                    if "manipulated_sequences" in root:
                        manipulated_paths.append(file_path)
                    elif "original_sequences" in root:
                        original_paths.append(file_path)
        
        with tqdm(len(manipulated_paths+original_paths)) as bar:
            for file_path in manipulated_paths:
                image_path = os.path.join(destination_path, "fake")
                filename = os.path.basename(file_path)
                if not os.path.exists(image_path):
                    os.makedirs(image_path)
                process_video(file_path, filename, image_path, num_frames, img_size)
                bar.update()
            for file_path in original_paths:
                image_path = os.path.join(destination_path, "real")
                filename = os.path.basename(file_path)
                if not os.path.exists(image_path):
                    os.makedirs(image_path)
                process_video(file_path, filename, image_path, num_frames, img_size)
                bar.update()
                
def process_df_files(dir_path, num_frames=1, img_size=(224, 224)):
    if os.path.isdir(dir_path):
        destination = list()
        list_ = list()
        data_set_splits = list()
        
        for data_folder in ["train", "val", "test"]:
            destination_paths = os.path.join("data", data_folder)
            list_paths = os.path.join("lists", "lists_df_1", "splits", data_folder+".txt")
            destination.append(destination_paths)
            list_.append(list_paths)
        
        for destination_paths in destination:
            if not os.path.exists(destination_paths):
                os.makedirs(destination_paths)
                
        for list_dataset in list_:
            dataset = filter_df_files(dir_path, list_dataset)
            data_set_splits.append(dataset)
                
        for dataset, destination in zip(data_set_splits, destination):
            if destination == os.path.join("data", "val"):
                process_df_videos(dataset, destination, num_frames, img_size)
            else:
                process_df_videos(dataset, destination, num_frames, img_size)
                                  
                
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
            #img = Image.fromarray(face)
            #img.show()
            for face_location in face_locations:
                top, right, bottom, left = face_location
                face_image = face[top:bottom, left:right]
                #img = Image.fromarray(face)
                #img.show()
                resized_face = cv2.resize(face_image, img_size, interpolation=cv2.INTER_AREA)
                resized_face = cv2.cvtColor(resized_face, cv2.COLOR_RGB2BGR)
                
                destination_path = os.path.join(image_path, filename[:-4])
                cv2.imwrite(destination_path+"_"+str(n)+".jpg", resized_face, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                
                n += 1
                
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="DFDC", type=str, help='Dataset (DFDC / FACEFORENSICS/ CELEB-DF / DF)')
    parser.add_argument('--raw_data_path', default='', type=str, help='Raw Videos directory')
    opt = parser.parse_args()
    
    if opt.dataset.upper() == "DFDC":
        print("Process DFDC data...")
        start_time = perf_counter()
        preprocess_dfdc_files(opt.raw_data_path)
        end_time = perf_counter()
        print("--- %s seconds ---" % (end_time - start_time))
    elif opt.dataset.upper() == "CELEB-DF":
        print("Process CELEB-DF data...")
        start_time = perf_counter()
        preprocess_celeb_files(opt.raw_data_path)
        end_time = perf_counter()
        print("--- %s seconds ---" % (end_time - start_time))
    elif opt.dataset.upper() == "FACEFORENSICS":
        print("Process FACEFORENSICS data...")
        start_time = perf_counter()
        preprocess_ff_files(opt.raw_data_path)
        end_time = perf_counter()
        print("--- %s seconds ---" % (end_time - start_time))
    elif opt.dataset.upper() == "DF":
        print("Process DEEPERFORENSICS data...")
        start_time = perf_counter()
        process_df_files(opt.raw_data_path)
        end_time = perf_counter()
        print("--- %s seconds ---" % (end_time - start_time))
                       
if __name__ == "__main__":
    main()