import argparse
import matplotlib.pyplot as plt

def read_loss_log(path):
    epochs = []
    losses = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ep, loss = line.split(",")
                epochs.append(int(ep))
                losses.append(float(loss))
            except ValueError:
                continue

    return epochs, losses


def main():
    parser = argparse.ArgumentParser(description="Plot training loss curve")
    parser.add_argument("--log", type=str, default="loss_log.txt",
                        help="Path to loss log file")
    parser.add_argument("--out", type=str, default="loss_curve.png",
                        help="Output image file")

    args = parser.parse_args()

    epochs, losses = read_loss_log(args.log)

    if len(epochs) == 0:
        print("No valid data found in log file.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.out)
    plt.show()

    print(f"Saved loss plot to {args.out}")


if __name__ == "__main__":
    main()
