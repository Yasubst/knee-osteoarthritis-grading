
import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold


dataset_path = "data/archive/Labeled"
labels = ["0", "1", "2", "3", "4"]

image_data = []

for label in labels:
    label_path = os.path.join(dataset_path, label)

    for image_file in sorted(os.listdir(label_path)):
        if image_file.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(label_path, image_file)
            image_data.append((img_path, label))

df = pd.DataFrame(image_data, columns=["image_path", "label"])

print(df.head())
print("Total images:", len(df))

# Split data into 80% development and 20% testin
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

# Create folder for CSV files
os.makedirs("data/CSVs", exist_ok=True)

# Save the test set
test_df.to_csv("data/CSVs/test.csv", index=False)

print("Development images:", len(train_df))
print("Test images:", len(test_df))


# Create 5-fold cross-validation
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

for fold, (train_idx, val_idx) in enumerate(
    skf.split(train_df, train_df["label"]), start=1
):

    fold_train = train_df.iloc[train_idx]
    fold_val = train_df.iloc[val_idx]

    fold_train.to_csv(
        f"data/CSVs/train_fold{fold}.csv", index=False
    )

    fold_val.to_csv(
        f"data/CSVs/val_fold{fold}.csv", index=False
    )

    print(
        f"Fold {fold}: Train = {len(fold_train)}, "
        f"Validation = {len(fold_val)}"
    )