
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import balanced_accuracy_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize

from args import get_args


args = get_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_metrics(labels, probabilities):
    labels = np.array(labels)
    probabilities = np.array(probabilities)

    predictions = probabilities.argmax(axis=1)
    binary_labels = label_binarize(labels, classes=range(args.num_classes))

    balanced_acc = balanced_accuracy_score(labels, predictions)
    roc_auc = roc_auc_score(binary_labels, probabilities, average="macro")
    avg_precision = average_precision_score(binary_labels, probabilities, average="macro")

    return balanced_acc, roc_auc, avg_precision


def train_model(model, train_loader, val_loader, fold):
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd
    )

    best_balanced_acc = -1.0
    os.makedirs(args.output_dir, exist_ok=True)

    metrics = ["Loss", "Balanced Accuracy", "ROC-AUC", "Average Precision"]
    train_history = {metric: [] for metric in metrics}
    val_history = {metric: [] for metric in metrics}

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        train_labels = []
        train_probabilities = []

        for batch in train_loader:
            xray_images = batch["xray"].float().to(device)
            labels = batch["target"].long().to(device)

            xray_images = xray_images.unsqueeze(1).repeat(1, 3, 1, 1) / 255.0

            optimizer.zero_grad()
            outputs = model(xray_images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_labels.extend(labels.cpu().tolist())
            train_probabilities.extend(
                torch.softmax(outputs.detach(), dim=1).cpu().tolist()
            )

        train_loss = running_loss / len(train_loader)
        train_bal_acc, train_auc, train_ap = calculate_metrics(
            train_labels, train_probabilities
        )

        val_loss, val_bal_acc, val_auc, val_ap = validate_model(
            model, val_loader, criterion
        )

        train_values = [train_loss, train_bal_acc, train_auc, train_ap]
        val_values = [val_loss, val_bal_acc, val_auc, val_ap]

        for metric, train_value, val_value in zip(metrics, train_values, val_values):
            train_history[metric].append(train_value)
            val_history[metric].append(val_value)

        if val_bal_acc > best_balanced_acc:
            best_balanced_acc = val_bal_acc
            torch.save(
                model.state_dict(),
                os.path.join(args.output_dir, f"best_model_fold_{fold}.pth")
            )

        print(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"Train Loss: {train_loss:.4f} - "
            f"Val Loss: {val_loss:.4f} - "
            f"Val Balanced Acc: {val_bal_acc:.4f} - "
            f"Val ROC-AUC: {val_auc:.4f} - "
            f"Val AP: {val_ap:.4f}"
        )

    for metric in metrics:
        plt.figure()
        plt.plot(train_history[metric], label="Training")
        plt.plot(val_history[metric], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(f"Fold {fold} - {metric}")
        plt.legend()
        plt.tight_layout()

        filename = metric.lower().replace(" ", "_")
        plt.savefig(
            os.path.join(args.output_dir, f"fold_{fold}_{filename}.png")
        )
        plt.close()

    return model


def validate_model(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for batch in val_loader:
            xray_images = batch["xray"].float().to(device)
            labels = batch["target"].long().to(device)

            xray_images = xray_images.unsqueeze(1).repeat(1, 3, 1, 1) / 255.0

            outputs = model(xray_images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            all_labels.extend(labels.cpu().tolist())
            all_probabilities.extend(
                torch.softmax(outputs, dim=1).cpu().tolist()
            )

    val_loss = running_loss / len(val_loader)
    balanced_acc, roc_auc, avg_precision = calculate_metrics(
        all_labels, all_probabilities
    )

    return val_loss, balanced_acc, roc_auc, avg_precision

