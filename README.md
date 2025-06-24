# SegRExtNet: A Deep Learning-based road segmentation

This project implements a deep learning model for aerial image segmentation, particularly focusing on roads detection using CNN-based architectures.

---

## 📁 Project Structure

├── checkpoints/ # Saved model weights
├── dataset/ # Your dataset (train, val, test)
│ ├── train/image/
│ ├── train/mask/
│ ├── val/image/
│ ├── val/mask/
│ ├── test/image/
│ └── test/mask/
├── outputs/ # Predicted masks from test.py
├── logs/ # Training logs
├── train.py # Training script
├── test.py # Testing/evaluation script
├── model.py # Model architecture (UNet/CompNet)
├── model_mnetv3_2_ca_sa.py # MobileNetV3 + CA + SA model
├── loss.py # Standard and custom loss functions
├── Hybrid_Eloss.py # Custom loss function
├── utils.py # Helper functions
├── data.py # Dataset loading
├── requirements.txt # Required Python packages
└── README.md # This file

## ⚙️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/hlmhlr/Road-segmentation.git
cd road_segmentation
pip install -r requirements.txt

2. **Install dependencies**

```bash
pip install -r requirements.txt

✅ Tip: Use a virtual environment:

```bash
conda create -n segrext python=3.8
conda activate segrext


## 🚀 Usage

▶️ Training the Model

python train.py \
    --path ./dataset \
    --checkpoint_path ./checkpoints/best_model.pth \
    --train_log_path ./logs/train_log.txt \
    --batch_size 4 \
    --num_epochs 40



Arguments:

--path: Root path to the dataset.

--checkpoint_path: Where to save the trained .pth model.

--train_log_path: File to log loss and metrics.

--batch_size: Batch size for training.

--num_epochs: Number of training epochs.



## 🧪 Testing / Inference

python test.py \
    --path ./dataset \
    --checkpoint_path ./checkpoints/best_model.pth \
    --results_path ./outputs

Arguments:

--path: Dataset path (should include test/image and test/mask).

--checkpoint_path: Path to the trained model file.

--results_path: Folder to save the predicted masks.


## 🧾 Dataset Format
The dataset must follow this structure:

dataset/
├── train/
│   ├── image/
│   └── mask/
├── val/
│   ├── image/
│   └── mask/
└── test/
    ├── image/
    └── mask/

All images are kept as .png files.

## 📈 Output & Metrics
At the end of testing, the following segmentation metrics are printed and saved:

Jaccard Index (IoU)

F1 Score

Recall

Precision

Accuracy

F2 Score

Inference FPS (Frames per Second)

Outputs:
Predicted masks are saved to the outputs/ folder.

Evaluation results are stored in a .csv file inside the results directory.

## 📌 Notes
Images are resized to 256×256 before inference.

Predicted masks are binarized with a threshold of 0.5.

The model is evaluated on test.py using ground-truth masks.

The entire code supports CUDA if a GPU is available.


---