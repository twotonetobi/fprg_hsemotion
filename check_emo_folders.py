import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch
import pandas as pd
import timm
from facenet_pytorch import MTCNN
from hsemotion.facial_emotions import HSEmotionRecognizer

model_name = 'enet_b0_8_best_afew'
# other models available
# model_name='enet_b0_8_best_vgaf'
# model_name='enet_b0_8_va_mtl'
# model_name='enet_b2_8'

def detect_face(frame):
    bounding_boxes, probs = mtcnn.detect(frame, landmarks=False)
    # use 0.99 for pictures that are filled with a face
    bounding_boxes = bounding_boxes[probs > 0.99]
    return bounding_boxes


def folder_emotion(path,annex):
    # Get the list of filenames (ignoring files that start with '.')
    filenames = [f for f in os.listdir(path) if not f.startswith('.')]
    total_files = len(filenames)
    data_list = []

    for i, filename in enumerate(filenames):
        fpath = os.path.join(path, filename)

        frame_bgr = cv2.imread(fpath)
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        emotion, scores = fer.predict_emotions(frame, logits=True)

        # Convert scores to a list if it's not already
        if not isinstance(scores, list):
            scores = scores.tolist()

        # Append the results as a dictionary
        data = {'filename': filename, 'emotion': emotion}
        data.update({f'score_{i + 1}': score for i, score in enumerate(scores)})
        data_list.append(data)

        # Print progress every 1000 files
        if (i + 1) % 1000 == 0:
            print(f'Processed orig {i + 1} files ({(i + 1) / total_files * 100:.2f}% done)')

    # Define the column order
    cols = ['filename', 'emotion'] + [f'score_{i + 1}' for i in range(8)]

    # Create the DataFrame from the list of dictionaries
    results = pd.DataFrame(data_list, columns=cols)
    results = results.add_suffix(annex)

    # Create a path for your output csv and pkl file
    csv_path = f'outputs/results{annex}.csv'
    pickle_file_path = f'outputs/results{annex}.pkl'
    results.to_csv(csv_path, index=False)
    results.to_pickle(pickle_file_path)


if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=40, device=device)
    # print(use_cuda)
    fer = HSEmotionRecognizer(model_name=model_name, device=device)

    # check all the generated folders and generate the corresponding csv and pkl files for data analysis
    # adopt for your path
    path = '/media/herbie/Tamarinde/original_images2/'
    annex = '_orig'
    folder_emotion(path, annex)

    path = '/media/herbie/Tamarinde/restored_images_face_recon_gfpgan2/'
    annex = '_recon_gfpgan'
    folder_emotion(path, annex)
    

    ```
    path = '/media/herbie/Tamarinde/restored_images_pix2pix2_gfpgan2/'
    annex = '_pix2pix_gfpgan'
    folder_emotion(path, annex)

    path = '/media/herbie/Tamarinde/restored_images_face_recon2/'
    annex = '_recon'
    folder_emotion(path, annex)

    path = '/media/herbie/Tamarinde/restored_images_pix2pix2/'
    annex = '_pix2pix'
    folder_emotion(path, annex)
    
    path = '/media/herbie/Tamarinde/restored_images_face_recon2_codeformer2/'
    annex = '_recon_codeformer'
    folder_emotion(path, annex)

    path = '/media/herbie/Tamarinde/restored_images_pix2pix2_codeformer2/'
    annex = '_pix2pix_codeformer'
    folder_emotion(path, annex)
    
    path = '/media/herbie/Tamarinde/meta_pro_avatar/'
    annex = '_meta_pro_avatar'
    folder_emotion(path, annex)

    path = '/media/herbie/Tamarinde/meta_pro_avatar/'
    annex = '_meta_pro_real_pix2pix_gfpgan'
    folder_emotion(path, annex)

    path = '/media/herbie/Tamarinde/meta_pro_avatar/'
    annex = '_meta_pro_real_pix2pix_codeformer'
    folder_emotion(path, annex)
    ```
