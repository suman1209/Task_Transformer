#### TASK ####
# The task is to predict ordered corners of the polygon given an input image. The data can be found in "data" directory with train val split.
# Implement the dataset loader, model and loss.
# Don't change the config parameters.
# The model visualises and eval every 50 epochs. Send the code for the final solution.
# It is a difficult setup, complete accuracy is not expected but atleast highest mIOU of "0.4" is required.
# The solution should be implemented based on transformer or diffusion models.
# Max permisible model size is 5M parameters.
#### TASK ####

import os
import pickle as pkl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

# ----- Config - DO NOT CHANGE THIS ----- #
EPOCHS = 800
BATCH_SIZE = 32
LR = 2e-4
IMG_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----- Config - DO NOT CHANGE THIS ----- #

# ----- Utils - DO NOT CHANGE THIS ------ #
def compute_polygon_iou(pred_coords, gt_coords):
    pred_poly = Polygon(pred_coords)
    gt_poly = Polygon(gt_coords)

    if not pred_poly.is_valid:
        return 0.0

    inter_area = pred_poly.intersection(gt_poly).area
    union_area = pred_poly.union(gt_poly).area

    if union_area == 0:
        return 0.0
    return inter_area / union_area

def plot_and_iou(img, pred_coords, gt_coords, save_path=None):
    pred_coords = pred_coords.detach().cpu().numpy()
    gt_coords = gt_coords.cpu().numpy()
    img_np = img.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    axes[1].imshow(np.ones_like(img_np) * 255)
    gt_coords = np.vstack([gt_coords, gt_coords[0:1]])
    axes[1].plot(gt_coords[:, 0], gt_coords[:, 1], 'go--', label='GT')

    iou = compute_polygon_iou(pred_coords, gt_coords)
    if len(pred_coords) >= 2:
        pred_coords = np.vstack([pred_coords, pred_coords[0:1]])
    axes[1].plot(pred_coords[:, 0], pred_coords[:, 1], 'ro--', label='Pred')

    axes[1].set_title(f"GT vs Pred | IOU: {iou:.2f}")
    axes[1].legend()
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    return iou

def plot_eval_curve(eval_dict):
    x = list(eval_dict.keys())
    y = list(eval_dict.values())
    plt.plot(x, y, marker='o', linestyle='-', color='steelblue', linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2)
    plt.title("mIoU")
    save_path = f"predictions/eval.png"
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(save_path, dpi=300)
    plt.close()

def check_model_size(model):
    total_params = sum(p.numel() for p in model.parameters()) / 1_000_000
    assert total_params < 5, f"Model size exceeds 5M params! Current size: {total_params:.2f}M"
# ----- Utils - DO NOT CHANGE THIS ------ #

# ----- Dataset ----- #
class PolygonDataset(Dataset):
    def __init__(self, pkl_file, transform=None):
        with open(pkl_file, "rb") as f:
            data = pkl.load(f)
        self.images = data["data"]
        self.targets = data["target"]

        # ADD CODE HERE #
        # ADD CODE HERE #

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # ADD CODE HERE #
        # ADD CODE HERE #
        pass

# ----- Collate Function ----- #
def collate_fn(batch):
    # ADD CODE HERE #
    # ADD CODE HERE #
    pass


# ----- Model -----
class PolygonModel(nn.Module):
    # ADD CODE HERE #
    # ADD CODE HERE #
    pass

# ----- Loss -----
def compute_loss(pred_coords, pred_exists, gt_coords, lengths):
    # ADD CODE HERE #
    # ADD CODE HERE #
    pass


# ----- Main -----
def main():
    print(f"Using device: {DEVICE}")
    os.makedirs("predictions", exist_ok=True)

    train_ds = PolygonDataset("data/train.pkl")
    val_ds = PolygonDataset("data/val.pkl")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, collate_fn=collate_fn)

    model = PolygonModel().to(DEVICE)
    check_model_size(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    eval_dict = dict()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.

        # ADD CODE HERE #
        for imgs, _ in train_loader:
            
            loss = compute_loss()
            # ADD CODE HERE #

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        scheduler.step()
        print(f"[Epoch {epoch+1}] Total Loss: {total_loss:.4f}", flush=True)

        # eval and vis
        if (epoch+1) % 50 == 0:
            miou = 0
            model.eval()
            with torch.no_grad():
                for i, (imgs, coords, lengths) in enumerate(val_loader):
                    imgs = imgs.to(DEVICE)
                    coords = [c.to(DEVICE) for c in coords]
                    
                    # ADD CODE HERE #
                    pred_coords = None
                    # ADD CODE HERE #

                    save_path = f"predictions/{i:03d}.png"
                    miou += plot_and_iou(imgs[0], pred_coords[0], coords[0], save_path=save_path)

            eval_dict[epoch] = miou / len(val_loader)

            plot_eval_curve(eval_dict)

            model.train()

if __name__ == "__main__":
    main()
