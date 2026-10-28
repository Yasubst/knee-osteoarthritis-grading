
import os
import pandas as pd
from torch.utils.data import DataLoader

from args import get_args
from dataset import knee_xray_dataset
from model import KneeOsteoarthritisModel
from trainer import train_model


args = get_args()

# 1. Iterate over each fold
for fold in range(1, args.num_folds + 1):
    print(f"Training Fold {fold}")

    train_csv = os.path.join(args.csv_dir, f"train_fold{fold}.csv")
    val_csv = os.path.join(args.csv_dir, f"val_fold{fold}.csv")

    train_set = pd.read_csv(train_csv)
    val_set = pd.read_csv(val_csv)

    # 2. Prepare datasets
    train_dataset = knee_xray_dataset(train_set)
    val_dataset = knee_xray_dataset(val_set)

    # 3. Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.bs, shuffle=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.bs, shuffle=False
    )

    # 4. Initialize the model
    model = KneeOsteoarthritisModel(
        backbone=args.backbone,
        pretrained=args.pretrained
    )

    # 5. Train the model
    train_model(model, train_loader, val_loader, fold)
