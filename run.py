# run.py
import argparse
import torch
from src import config
from src.dataset import get_dataloaders, save_stats
from src.model import build_model
from src.train import train_model
from src.evaluate import evaluate_model, collect_predictions, confusion_matrix_plot
from src.predict import predict_image

def main():
    parser = argparse.ArgumentParser(description="CNN Image Recognition CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Train
    train_parser = subparsers.add_parser("train", help="Train the CNN model")
    train_parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    train_parser.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    train_parser.add_argument("--lr", type=float, default=config.DEFAULT_LR)

    # Evaluate
    subparsers.add_parser("evaluate", help="Evaluate the trained model")

    # Predict
    predict_parser = subparsers.add_parser("predict", help="Predict a single image")
    predict_parser.add_argument("--image", type=str, required=True)

    # Plot confusion matrix
    plot_parser = subparsers.add_parser("plot", help="Plot confusion matrix")
    plot_parser.add_argument("--out", type=str, default="logs/confusion_matrix.png")

    args = parser.parse_args()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")

    # Load data
    loaders, classes = get_dataloaders(
        config.RAW_DIR,
        config.PROCESSED_DIR,
        batch_size=args.batch_size if hasattr(args, "batch_size") else config.DEFAULT_BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        val_size=5000,
        seed=config.SEED,
        pin_memory=True,
    )

    # Build model
    model = build_model(num_classes=len(classes)).to(device)

    if args.command == "train":
        cfg = {"epochs": args.epochs, "lr": args.lr, "weight_decay": config.DEFAULT_WEIGHT_DECAY}
        model, history = train_model(model, loaders, device, cfg)

    elif args.command == "evaluate":
        metrics = evaluate_model(model, loaders["test"], device)
        print(f"[Test] Loss: {metrics['loss']:.4f}, Accuracy: {metrics['acc']:.2f}%")

    elif args.command == "predict":
        result = predict_image(model, args.image, device, classes)
        print(f"[Predict] Class: {result['pred_class']} | Probabilities: {result['probs']}")

    elif args.command == "plot":
        y_true, y_pred = collect_predictions(model, loaders["test"], device)
        confusion_matrix_plot(y_true, y_pred, classes, out_path=config.LOGS_DIR / args.out)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()