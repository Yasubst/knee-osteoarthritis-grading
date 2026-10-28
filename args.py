
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Knee Osteoarthritis Grading"
    )

    parser.add_argument("-backbone", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50"])

    parser.add_argument("-pretrained", action="store_true")

    parser.add_argument("-output_dir", type=str, default="sessions")

    parser.add_argument("-csv_dir", type=str, default="data/CSVs")

    parser.add_argument("-epochs", type=int, default=50)

    parser.add_argument("-bs", type=int, default=16,
                        choices=[16, 32, 64, 128])

    parser.add_argument("-lr", type=float, default=1e-3,
                        choices=[1e-4, 1e-3])

    parser.add_argument("-num_classes", type=int, default=5)

    parser.add_argument("-num_folds", type=int, default=5)

    parser.add_argument("-data_dir", type=str,
                        default="data/archive/Labeled")

    parser.add_argument("-wd", type=float, default=0.0)

    parser.add_argument("-test", action="store_true")
    return parser.parse_args()
