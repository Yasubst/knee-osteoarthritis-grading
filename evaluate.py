
import os
import torch
import pandas as pd
from utils import plot_confusion_matrix

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report

from args import get_args
from dataset import knee_xray_dataset
from model import KneeOsteoarthritisModel


args = get_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_labels = []
all_predictions = []

folds = [4] if args.test else range(1, args.num_folds + 1)

for fold in folds:
    print(f"\nEvaluating Fold {fold}")

    val_csv = os.path.join(
    args.csv_dir,
    "test.csv" if args.test else f"val_fold{fold}.csv"
    )
    val_data = pd.read_csv(val_csv)

    val_dataset = knee_xray_dataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)

    model = KneeOsteoarthritisModel(
        backbone=args.backbone,
        pretrained=False
    ).to(device)

    checkpoint = os.path.join(
    args.output_dir,
    "baseline" if args.test else "",
    f"{args.backbone}_pretrained_fold{fold}.pth"
    )

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    fold_labels = []
    fold_predictions = []

    with torch.no_grad():
        for batch in val_loader:
            xray_images = batch["xray"].float().to(device)
            labels = batch["target"].long().to(device)

            xray_images = xray_images.unsqueeze(1)
            xray_images = xray_images.repeat(1, 3, 1, 1) / 255.0

            outputs = model(xray_images)
            predictions = outputs.argmax(dim=1)

            fold_labels.extend(labels.cpu().tolist())
            fold_predictions.extend(predictions.cpu().tolist())

    accuracy = accuracy_score(fold_labels, fold_predictions)
    balanced_accuracy = balanced_accuracy_score(fold_labels, fold_predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")

    all_labels.extend(fold_labels)
    all_predictions.extend(fold_predictions)

print("\nFinal Test Results" if args.test else "\nOverall Cross-Validation Results")
print(f"Accuracy: {accuracy_score(all_labels, all_predictions):.4f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(all_labels, all_predictions):.4f}")

print("\nClassification Report:")
print(classification_report(
    all_labels, all_predictions,
    labels=[0, 1, 2, 3, 4],
    target_names=["KL0", "KL1", "KL2", "KL3", "KL4"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_predictions, labels=[0, 1, 2, 3, 4]))

if args.test:
    plot_confusion_matrix(all_labels, all_predictions)
