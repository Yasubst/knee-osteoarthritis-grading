import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(labels, predictions):
    ConfusionMatrixDisplay.from_predictions(
        labels,
        predictions,
        labels=[0, 1, 2, 3, 4],
        display_labels=[
    "KL0\nNone",
    "KL1\nDoubtful",
    "KL2\nMild",
    "KL3\nModerate",
    "KL4\nSevere"
    ],
        cmap="Blues"
    )

    plt.title("Knee Osteoarthritis Grading - Confusion Matrix")
    plt.tight_layout()
    plt.savefig("sessions/confusion_matrix.png", dpi=300)
    plt.close()