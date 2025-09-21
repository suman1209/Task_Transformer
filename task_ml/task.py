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
from torchvision import models
from scipy.optimize import linear_sum_assignment as lsa

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
        self.transform = transform if transform else transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])
        self.images = [self.transform(img) for img in self.images]
        self.targets = [torch.tensor(gt, dtype=torch.float32) / IMG_SIZE
                        for gt in self.targets]
        # ADD CODE HERE #

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # ADD CODE HERE #
        return self.images[idx], self.targets[idx]
        # ADD CODE HERE #
        pass

# ----- Collate Function ----- #
def collate_fn(batch):
    # ADD CODE HERE #
    imgs, coords = zip(*batch)
    imgs = torch.stack([img for img in imgs], dim=0)
    lengths = [len(c) for c in coords]
    return imgs, coords, lengths
    # ADD CODE HERE #
    pass


# ----- Model -----
class PolygonModel(nn.Module):
    # ADD CODE HERE #
    def __init__(self):
        super(PolygonModel, self).__init__()
        self.d_model = 64
        self.feature_extractor = models.squeezenet1_0(pretrained=True).features
        # since given images are grayscale (1 channel), conversion to RGB 
        self.gray_to_rgb = nn.Conv2d(1, 3, kernel_size=1)
        self.channel_reducer = nn.Conv2d(512, self.d_model, kernel_size=1)
        self.positional_encoding = nn.Parameter(torch.zeros(49, self.d_model))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                        nhead=8)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)

        # Learnable queries
        self.queries = nn.Parameter(torch.randn(4, self.d_model))
        # Decoder layer
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model,
                                                        nhead=8)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=6)

        # MLP heads 
        # Predict x, y coordinates
        self.coord_head = nn.Linear(self.d_model, 2)  
        # Predict presence/absence
        self.pred_exists_head = nn.Sequential(
            nn.Linear(self.d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B = x.size(0)
        # Convert grayscale to RGB
        x = self.gray_to_rgb(x)
        
        img_features = self.feature_extractor(x)
        img_features = self.channel_reducer(img_features)
        # Squash the spatial dimensions to form a sequence input for transfomer
        # (batch_size, sequence_length(spatial dimensions), input_dim(channels)
        B, C, H, W = img_features.shape
        tokenized_img_feats = img_features.view(B, H * W, C)
        # add positional encoding
        tokenized_img_feats += self.positional_encoding.unsqueeze(0)
        # transformed_img_features
        trans_img_feats = self.encoder(tokenized_img_feats).permute(1, 0, 2)

        # Expand queries for the batch (batch_size, 4, 512)
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)
        queries = queries.permute(1, 0, 2)
        # Decode using the queries and encoder output
        trans_queries = self.decoder(queries, trans_img_feats).permute(1, 0, 2)
        # Predict x, y coordinates
        # Shape: (batch_size, 4, 2)
        pred_coords = self.coord_head(trans_queries)
        # Predict presence/absence
        # Shape: (batch_size, 4, 1)
        pred_exists = self.pred_exists_head(trans_queries)
        # Apply sorting to predicted points
        pred_coords_sorted = self.sort_points_clockwise_batched(pred_coords)

        return pred_coords_sorted, pred_exists
    
    def sort_points_clockwise_batched(self, coords):
        """
        Sort points in clockwise order for each batch to form proper polygons.
        Args:
            coords: Tensor (batch_size, num_points, 2).
        Returns:
            Tensor (batch_size, num_points, 2), sorted in clockwise order.
        """

        # Compute the centroid for each batch
        # Shape: (batch_size, 1, 2)
        centroids = coords.mean(dim=1, keepdim=True)
        # Compute angles of each point relative to the centroid
        angles = torch.atan2(coords[:, :, 1] - centroids[:, :, 1],
                             coords[:, :, 0] - centroids[:, :, 0]) 
        # Sort points by angle for each batch
        sorted_indices = torch.argsort(angles, dim=1)
        sorted_coords = torch.gather(coords, dim=1,
                                     index=sorted_indices.unsqueeze(-1).expand(-1, -1, 2))

        return sorted_coords

    # ADD CODE HERE #
    pass

# ----- Loss -----
def compute_loss(pred_coords, pred_exists, gt_coords, lengths):
    # ADD CODE HERE #
    """
    Compute the loss for the model.
    Args:
        pred_coords: Tensor of shape (batch_size, num_queries, 2), predicted coordinates.
        pred_exists: Tensor of shape (batch_size, num_queries, 1), predicted presence/absence.
        gt_coords: List of Tensors, each of shape (num_gt_points, 2), ground truth coordinates.
        lengths: List of integers, number of ground truth points for each sample in the batch.
    Returns:
        total_loss: Combined loss (coordinate loss + presence loss).
    """
    batch_size, num_queries, _ = pred_coords.shape
    coord_loss = 0.0
    presence_loss = 0.0

    for b in range(batch_size):
        # Extract predictions and ground truth for the current batch
        pred_coords_b = pred_coords[b]  # Shape: (num_preds, 2)
        pred_exists_b = pred_exists[b].squeeze(-1)  # Shape: (num_queries,)
        gt_coords_b = gt_coords[b] 
        # Reshape ground truth into (num_gt_points, 2)
        gt_coords_b = gt_coords_b.view(-1, 2)  # Shape: (num_gt_points, 2)
        num_gt_points = lengths[b]  # Number of ground truth points
        # Compute cost matrix for matching (Euclidean distance)
        # Shape: (num_pred, num_gt_points)
        cost_matrix = torch.cdist(pred_coords_b, gt_coords_b, p=2)

        if torch.isnan(cost_matrix).any() or torch.isinf(cost_matrix).any():
            print("Invalid values detected in cost_matrix!")
            print(cost_matrix)
            breakpoint()
            raise ValueError("Cost matrix contains NaN or Inf values.")

        # Solve the assignment problem using the Hungarian algorithm
        row_indices, col_indices = lsa(cost_matrix.cpu().detach().numpy())
        
        # Match predictions to ground truth
        matched_preds = pred_coords_b[row_indices]  # Shape: (num_gt_points, 2)
        matched_gts = gt_coords_b[col_indices]  # Shape: (num_gt_points, 2)

        # Coordinate loss (MSE between matched predictions and ground truth)
        coord_loss += F.mse_loss(matched_preds, matched_gts)

        # Presence loss (MSE between predicted presence sum and gt length)
        gt_len = torch.tensor(num_gt_points, device=pred_exists.device,
                              dtype=torch.float)
        presence_loss += F.mse_loss(pred_exists_b.sum(0), gt_len)

    # Average losses over the batch
    coord_loss /= batch_size
    presence_loss /= batch_size

    # Combine the losses
    total_loss = coord_loss + 0.1*presence_loss
    return total_loss
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

        for imgs, gt_coords, lengths in train_loader:
            imgs = imgs.to(DEVICE)
            gt_coords = [c.to(DEVICE) for c in gt_coords]
            pred_coords, pred_exists = model(imgs)
            loss = compute_loss(pred_coords, pred_exists, gt_coords, lengths)
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
                    
                    coords = [co*IMG_SIZE for co in coords]
                    pred_coords, pred_exists = model(imgs)
                    pred_coords *= IMG_SIZE
                    coords = [co.view(-1, 2) for co in coords]
                    # ADD CODE HERE #

                    save_path = f"predictions/{i:03d}.png"
                    miou += plot_and_iou(imgs[0], pred_coords[0], coords[0], save_path=save_path)

            eval_dict[epoch] = miou / len(val_loader)

            plot_eval_curve(eval_dict)

            model.train()

if __name__ == "__main__":
    main()
