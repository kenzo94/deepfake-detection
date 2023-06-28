# Code from: https://github.com/erprogs/CViT/tree/main, https://github.com/selimsef/dfdc_deepfake_challenge access: 16.06.2023 15:00

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
from mtcnn import MTCNN
import matplotlib.pyplot as plt


videos_to_skip_train = [
        "030_W008.mp4", "069_W013.mp4", "045_W010.mp4", "052_W012.mp4", "108_W019.mp4",
        "112_W019.mp4", "170_W029.mp4", "176_W133.mp4", "190_W032.mp4", "193_W032.mp4",
        "284_W116.mp4", "287_W125.mp4", "295_W100.mp4", "299_M105.mp4", "355_W012.mp4",
        "397_W017.mp4", "410_W019.mp4", "411_W019.mp4", "426_W021.mp4", "430_W022.mp4",
        "446_W024.mp4", "450_W024.mp4", "479_W029.mp4", "507_W134.mp4", "555_W040.mp4",
        "575_W042.mp4", "602_W125.mp4", "603_W125.mp4", "607_W100.mp4", "627_M121.mp4",
        "652_W010.mp4", "662_W012.mp4", "695_M005.mp4", "697_M005.mp4", "706_W017.mp4",
        "725_W021.mp4", "726_W021.mp4", "733_W021.mp4", "743_W023.mp4", "748_W023.mp4",
        "786_W133.mp4", "831_W036.mp4", "889_W042.mp4", "892_W110.mp4", "908_M030.mp4",
        "935_W125.mp4", "948_M105.mp4", "962_W007.mp4", "965_W007.mp4", "373_M026.mp4",
        "475_W028.mp4", "950_M105.mp4"
    ]
videos_to_skip_val =[
        "263_W111.mp4", "265_W111.mp4", "291_M018.mp4", "418_W132.mp4", "459_W025.mp4",
        "586_W111.mp4", "667_W005.mp4", "724_W132.mp4", "904_W111.mp4"
    ]
videos_to_skip_test =[
        "075_W014.mp4", "386_W015.mp4", "223_W037.mp4", "099_W018.mp4", "422_M031.mp4"
        "460_W026.mp4", "508_W135.mp4", "516_W135.mp4", "569_M115.mp4", "581_M117.mp4",
        "678_W014.mp4", "683_W014.mp4", "713_W018.mp4", "874_M025.mp4", "894_M027.mp4",
        "460_W026.mp4"
    ]
        
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
            if any(video in line for video in videos_to_skip_train):
                continue
            if any(video in line for video in videos_to_skip_test):
                continue
            if any(video in line for video in videos_to_skip_val):
                continue
            
            fakes.append(os.path.join(path_raw[0], line))
            real = re.sub(r"_.*\.mp4", ".mp4", line)
            reals.append(os.path.join(path_raw[1], real))
    
    dataset["fakes"] = fakes
    dataset["reals"] = reals
        
    return dataset

def filter_df_files_rest(dir_path):
    dataset = dict()
    path_raw = []
   
    for dir in os.listdir(dir_path):
        path = os.path.join(dir_path, dir)
        path_raw.append(path)
        path_raw.sort()
        
    dataset["train_fakes"] = [os.path.join(path_raw[0], video) for video in videos_to_skip_train]
    dataset["train_real"] = [os.path.join(path_raw[1], re.sub(r"_.*\.mp4", ".mp4", video)) for video in videos_to_skip_train]
    dataset["val_fakes"] = [os.path.join(path_raw[0], video) for video in videos_to_skip_val]
    dataset["val_real"] = [os.path.join(path_raw[1], re.sub(r"_.*\.mp4", ".mp4", video)) for video in videos_to_skip_val]
    dataset["test_fakes"] = [os.path.join(path_raw[0], video) for video in videos_to_skip_test]
    dataset["test_real"] = [os.path.join(path_raw[1], re.sub(r"_.*\.mp4", ".mp4", video)) for video in videos_to_skip_test]
    
    return dataset

def process_df_videos_rest(dataset, key, destination, num_frames, img_size, seed):
    
    if "fake" in key:
        dir_ = "fake"
    elif "real" in key:
        dir_ = "real"
    
    print(f"Process {dir_}")
    with tqdm(len(dataset)) as bar:
        for video_path in dataset:
            image_path = os.path.join(destination, dir_)
            filename = os.path.basename(video_path)
            if not os.path.exists(image_path):
                os.makedirs(image_path)
            print(f"FAKE path: {video_path} - name: {filename} - destination: {image_path}")
            process_video_rest(video_path, filename, image_path, num_frames, img_size, seed)
            bar.update()

def process_df_videos(dataset, destination, num_frames, img_size):
    print("{cnt} Files to Process".format(cnt=len(dataset["fakes"])))
    fake = dataset["fakes"]
    real = dataset["reals"]
    print("Process Fakes")
    with tqdm(len(dataset["fakes"])) as bar:
        for video_path in fake:
            image_path = os.path.join(destination, "fake")
            filename = os.path.basename(video_path)
            if not os.path.exists(image_path):
                os.makedirs(image_path)
            print(f"FAKE path: {video_path} - name: {filename} - destination: {image_path}")
            process_video(video_path, filename, image_path, num_frames, img_size)
            bar.update()
    print("Process Reals")
    print("{cnt} Files to Process".format(cnt=len(dataset["reals"])))
    with tqdm(len(dataset["reals"])) as bar:
        for video_path in real:
            image_path = os.path.join(destination, "real")
            filename = os.path.basename(video_path)
            if not os.path.exists(image_path):
                os.makedirs(image_path)
            print(f"REAL path: {video_path} - name: {filename} - destination: {image_path}")
            process_video(video_path, filename, image_path, num_frames, img_size)
            bar.update()
                
                
def preprocess_dfdc_files(dir_path, num_frames=20, img_size=(224, 224)):
    # iterate over DFDC dataset
    if os.path.isdir(dir_path):
        destination_path = os.path.join("data", "test_dfdc")
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)
            
        data = load_metadata(dir_path)
        if data:
            filtered_files = filter_dfdc_files(data, 25)
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
    

def preprocess_celeb_files(dir_path, num_frames=20, img_size=(224, 224)):
     if os.path.isdir(dir_path):
        destination_path = os.path.join("data", "test_celeb")
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)
        
        for dir in os.listdir(dir_path):
            path = os.path.join(dir_path, dir)
            if os.path.isdir(path):
                files = filter_celeb_files(os.listdir(path), 25)
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
                        
def preprocess_ff_files(dir_path, num_frames=20, img_size=(224, 224)):
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
                
def process_df_files(dir_path, num_frames=50, img_size=(224, 224), skip=False):
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
        
        if skip == False:        
            for list_dataset in list_:
                dataset = filter_df_files(dir_path, list_dataset)
                data_set_splits.append(dataset)
                
            for dataset, destination in zip(data_set_splits, destination):
                if destination == os.path.join("data", "val"):
                    process_df_videos(dataset, destination, num_frames, img_size=img_size)
                elif destination == os.path.join("data", "test"):
                    process_df_videos(dataset, destination, num_frames=20, img_size=img_size)
                elif destination == os.path.join("data", "train"):
                    process_df_videos(dataset, destination, num_frames, img_size)
                    
        elif skip == True:
            dataset = filter_df_files_rest(dir_path)
            
            for key in dataset.keys():
                if "test" in key:
                    destination = os.path.join("data", "test")
                    process_df_videos_rest(dataset[key], key, destination, num_frames=20, img_size=img_size, seed=42)
                elif "val" in key:
                    destination = os.path.join("data", "val")
                    process_df_videos_rest(dataset[key], key, destination, num_frames=20, img_size=img_size, seed=42)
                elif "train" in key:
                    destination = os.path.join("data", "train")
                    process_df_videos_rest(dataset[key], key, destination, num_frames=20, img_size=img_size, seed=42)
                    
                
def process_video_rest(video_path, filename, image_path, num_frames, img_size, seed):
    capture = cv2.VideoCapture(video_path)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count <= 0:
        capture.release()
        return
    
    detector = MTCNN()
    
    # Zufällige Frame-Indizes auswählen
    np.random.seed(seed)
    step = frame_count // num_frames
    start_offset = np.random.randint(0, step)
    frame_idxs = np.arange(start_offset, frame_count, step)
    frame_idxs = np.clip(frame_idxs, 0, frame_count - 1)
    
    n=0
    for frame_idx in frame_idxs:
        # Frame an der aktuellen Indexposition lesen
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = capture.read()
        
        if not ret:
            continue
        
        # Gesichter im aktuellen Frame erkennen
        faces = detector.detect_faces(frame)
        
        if len(faces) == 0:
            continue
        
        # Das beste Gesicht auswählen
        best_face = max(faces, key=lambda f: f['confidence'])
        bounding_box = best_face['box']
        x, y, w, h = bounding_box
        
        # Gesicht im Frame ausschneiden
        face_image = frame[y:y+h, x:x+w]
        
        resized_face = cv2.resize(face_image, img_size, interpolation=cv2.INTER_AREA)
        resized_face = cv2.cvtColor(resized_face, cv2.COLOR_RGB2BGR)
                
        output_path = os.path.join(image_path, f"{filename[:-4]}_{n}.jpg")
        
        plt.imsave(output_path, resized_face)
                
        n += 1 
    
    capture.release()

    
# access video
def process_video(video_path, filename, image_path, num_frames, img_size, seed):
    gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    facedet = BlazeFace().to(gpu)
    facedet.load_weights(os.path.join("preprocessing", "blazeface.pth"))
    facedet.load_anchors(os.path.join("preprocessing", "anchors.npy"))
    _ = facedet.train(False)
    
    from video_reader import VideoReader
    from face_extractor import FaceExtractor
     
    video_reader = VideoReader()
    video_read_fn = lambda x: video_reader.read_random_frames_interval(x, num_frames=num_frames, seed=seed)
    face_extractor = FaceExtractor(video_read_fn, facedet)
    
    faces = face_extractor.process_video(video_path) #get list of faces-bbox as numpy arrays with values 0-255 RGB
    # Only look at one face per frame.
    face_extractor.keep_only_best_face(faces)
    
    n = 0
    for frame_data in faces:
        for face in frame_data["faces"]:
            face_locations = face_recognition.face_locations(face) #further crop the face
            for face_location in face_locations:
                top, right, bottom, left = face_location
                face_image = face[top:bottom, left:right]
                #img = Image.fromarray(face)
                #img.show()
                resized_face = cv2.resize(face_image, img_size, interpolation=cv2.INTER_AREA)
                resized_face = cv2.cvtColor(resized_face, cv2.COLOR_RGB2BGR)
                
                destination_path = os.path.join(image_path, filename[:-4])
                cv2.imwrite(destination_path+"_"+str(n)+".jpg", resized_face)
                
                n += 1 
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="DFDC", type=str, help='Dataset (DFDC / FACEFORENSICS/ CELEB-DF / DF / DF-REST)')
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
    elif opt.dataset.upper() == "DF-REST":
        print("Process DEEPERFORENSICS data...")
        start_time = perf_counter()
        process_df_files(opt.raw_data_path, skip=True)
        end_time = perf_counter()
        print("--- %s seconds ---" % (end_time - start_time))
                       
if __name__ == "__main__":
    main()
    #process_video(r"raw\deeperforensics\end_to_end\030_W008.mp4", "030_W008.mp4", r"raw\test", num_frames=35, img_size=(224, 224))