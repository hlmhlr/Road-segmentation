# SegRExtNet: A Deep Learning-based road segmentation

This project implements a deep learning model for aerial image segmentation, particularly focusing on roads detection using CNN-based architectures.

## 📁 Project Structure

- `train.py`: Script to train the segmentation model.
- `test.py`: Script to evaluate the trained model.
- `model.py`: Definition of the segmentation model (e.g., SegRExNet).
- `loss.py`, `Hybrid_Eloss.py`: Custom loss functions used during training.
- `data.py`: Dataset loader.
- `utils.py`: Utility functions (seeding, image pre/post-processing, etc.).
- `dataset/`: Folder where training, validation, and testing data are stored.
- `outputs/`: Directory to save predicted masks.
- `checkpoints/`: Saved model weights.
- `logs/`: Training logs and metrics.
- `files/`: Additional files like result CSVs.

## ⚙️ Installation

```bash
git clone <your-repo-url>
cd road_segmentation
pip install -r requirements.txt