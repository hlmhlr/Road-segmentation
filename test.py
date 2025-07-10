import argparse
import torchvision.transforms as transforms
import os, time
import pandas as pd
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import imageio
import torch
from sklearn.metrics import (
    jaccard_score, f1_score, recall_score, precision_score, accuracy_score, fbeta_score)


from model import SegRExtNet
from utils import create_dir, seeding, make_channel_last


def load_data(dataset_path, split=0.2):
    
    train_x = sorted(glob(os.path.join(dataset_path,"train/image","*.png")))
    train_y = sorted(glob(os.path.join(dataset_path,"train/mask", "*.png")))
    
    test_x = sorted(glob(os.path.join(dataset_path,"test/image", "*.png")))
    test_y = sorted(glob(os.path.join(dataset_path,"test/mask", "*.png")))
    
    valid_x = sorted(glob(os.path.join(dataset_path,"val/image", "*.png")))
    valid_y = sorted(glob(os.path.join(dataset_path,"val/mask", "*.png")))
    
    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def calculate_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    y_pred = y_pred > 0.5
    y_pred = y_pred.reshape(-1)
    y_pred = y_pred.astype(np.uint8)

    y_true = y_true > 0.5
    y_true = y_true.reshape(-1)
    y_true = y_true.astype(np.uint8)

    ## Score
    score_jaccard = jaccard_score(y_true, y_pred, average='binary')
    score_f1 = f1_score(y_true, y_pred, average='binary')
    score_recall = recall_score(y_true, y_pred, average='binary')
    score_precision = precision_score(y_true, y_pred, average='binary', zero_division=1)
    score_acc = accuracy_score(y_true, y_pred)
    score_fbeta = fbeta_score(y_true, y_pred, beta=1.0, average='binary', zero_division=1)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc, score_fbeta]

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    parser = argparse.ArgumentParser(description="Run model inference and evaluation.")

    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--results_path', type=str, required=True, help='Path to save predicted result images')
    parser.add_argument('--csv_save_path', type=str, required=True, help='Path to save metrics CSV result file')

    args = parser.parse_args()



    """ Load dataset """
    path = args.dataset_path
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)
    
    

    """ Hyperparameters """
    size = (256, 256)

    if os.path.exists(args.results_path):
        print("Output folder exists")
    else:
        os.makedirs(args.results_path, exist_ok=True)





    """ Load the checkpoint """
    checkpoint_path = args.checkpoint_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SegRExtNet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    """ Testing """
    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []
    SCORE=[]


    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y))):
        name = y.split("/")[-1].split(".")[0]

        ## Image
        image = cv2.imread(x, cv2.IMREAD_COLOR)

        image1 = cv2.resize(image, size)
        ori_img1 = image1
        image1 = np.transpose(image1, (2, 0, 1))
        image1 = image1/255.0
        image1 = np.expand_dims(image1, axis=0)
        image1 = image1.astype(np.float32)
        image1 = torch.from_numpy(image1)
        image1 = image1.to(device)

        ## Mask
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)

        mask1 = cv2.resize(mask, size)
        ori_mask1 = mask1
        mask1 = np.expand_dims(mask1, axis=0)
        mask1 = mask1/255.0
        mask1 = np.expand_dims(mask1, axis=0)
        mask1 = mask1.astype(np.float32)
        mask1 = torch.from_numpy(mask1)
        mask1 = mask1.to(device)

        with torch.no_grad():
            """ FPS calculation """
            start_time = time.time()
            pred_y1 = torch.sigmoid(model(image1))
            end_time = time.time() - start_time
            time_taken.append(end_time)
            # print("{} - {:.10f}".format(name, end_time))

            """ Evaluation metrics """
            score = calculate_metrics(mask1, pred_y1)
            SCORE.append([name, score[0], score[1], score[2], score[3], score[4], score[5]])
            metrics_score = list(map(add, metrics_score, score))


            """ Predicted Mask """
            pred_y1 = pred_y1[0].cpu().numpy()
            pred_y1 = np.squeeze(pred_y1, axis=0)
            pred_y1 = pred_y1 > 0.5
            pred_y1 = pred_y1.astype(np.int32)
            pred_y1 = pred_y1 * 255
            pred_y1 = np.array(pred_y1, dtype=np.uint8)

        ori_img1 = ori_img1
        ori_mask1 = mask_parse(ori_mask1)
        pred_y1 = mask_parse(pred_y1)
        sep_line = np.ones((size[0], 10, 3)) * 255

        tmp = [
            ori_img1, sep_line,
            ori_mask1, sep_line,
            pred_y1
        ]
                                                                                                             
        cat_images = np.concatenate(tmp, axis=1)
        
        #dataset image save
        cv2.imwrite(os.path.join(args.results_path, f"{name}.png"), pred_y1)



    jaccard = metrics_score[0]/len(test_x)
    f1 = metrics_score[1]/len(test_x)
    recall = metrics_score[2]/len(test_x)
    precision = metrics_score[3]/len(test_x)
    acc = metrics_score[4]/len(test_x)
    f2 = metrics_score[5]/len(test_x)

    print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f} - F2: {f2:1.4f}")

    mean_time_taken = np.mean(time_taken)
    mean_fps = 1/mean_time_taken
    print("Mean FPS: ", mean_fps)
    
    df = pd.DataFrame(SCORE, columns = ["Image Name", "Jaccard" , "F1", "Recall", "Precision", "Acc", "F2"])
    df.to_csv(os.path.join(args.csv_save_path, "output.csv") )