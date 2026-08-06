"""Train the railway asset classifier.

    python model/train.py --epochs 30

Phase 1 trains the new head with the backbone frozen, phase 2 unfreezes
the last block and refines both at a lower learning rate.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from dataset import build_loaders
    from network import (
        CHECKPOINT_PATH,
        build_model,
        pick_device,
        save_checkpoint,
        unfreeze_last_block,
    )
else:
    from .dataset import build_loaders
    from .network import (
        CHECKPOINT_PATH,
        build_model,
        pick_device,
        save_checkpoint,
        unfreeze_last_block,
    )


HISTORY_PATH = Path(__file__).resolve().parent / "training_history.json"


def run_epoch(model, loader, criterion, device, optimiser=None):
    """One pass over a loader. Trains when given an optimiser."""
    training = optimiser is not None

    model.train() if training else model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            running_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        return 0.0, 0.0

    return running_loss / total, correct / total


def train_phase(model, loaders, device, epochs, learning_rate, label, state):
    """Run one training phase, keeping the best validation weights."""
    train_loader, val_loader = loaders

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    trainable = [p for p in model.parameters() if p.requires_grad]

    optimiser = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    print(f"\n{label} - {epochs} epochs, lr {learning_rate}, "
          f"{sum(p.numel() for p in trainable):,} trainable parameters")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimiser
        )

        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)

        scheduler.step()

        marker = ""

        if val_acc > state["best_val_acc"]:
            state["best_val_acc"] = val_acc

            # .cpu() so the copy survives regardless of where training ran
            state["best_weights"] = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            marker = "  <- best so far"

        state["history"].append({
            "phase": label,
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
        })

        print(
            f"  epoch {epoch:>2}/{epochs}  "
            f"train loss {train_loss:.3f} acc {train_acc:.1%}  |  "
            f"val loss {val_loss:.3f} acc {val_acc:.1%}{marker}"
        )

    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--epochs", type=int, default=15,
                        help="head-only epochs (default 15)")
    parser.add_argument("--finetune-epochs", type=int, default=10,
                        help="fine tuning epochs (default 10)")
    parser.add_argument("--no-finetune", action="store_true",
                        help="skip the fine tuning phase")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate for the head")
    parser.add_argument("--finetune-lr", type=float, default=1e-4,
                        help="learning rate once the backbone is unfrozen")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=CHECKPOINT_PATH)

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = pick_device()

    train_loader, val_loader, test_loader, classes = build_loaders(
        batch_size=args.batch_size,
        seed=args.seed
    )

    print(f"Device: {device}")
    print(f"Classes ({len(classes)}): {', '.join(classes)}")
    print(
        f"Images: {len(train_loader.dataset)} train, "
        f"{len(val_loader.dataset)} val, {len(test_loader.dataset)} test"
    )

    model = build_model(len(classes), freeze_backbone=True).to(device)

    state = {
        "best_val_acc": 0.0,
        "best_weights": None,
        "history": [],
    }

    started = time.time()

    train_phase(
        model,
        (train_loader, val_loader),
        device,
        args.epochs,
        args.lr,
        "Phase 1: head only",
        state
    )

    if not args.no_finetune:
        unfreeze_last_block(model)

        train_phase(
            model,
            (train_loader, val_loader),
            device,
            args.finetune_epochs,
            args.finetune_lr,
            "Phase 2: fine tuning layer4",
            state
        )

    # roll back to the best epoch before measuring or saving
    if state["best_weights"] is not None:
        model.load_state_dict(state["best_weights"])

    model.to(device)

    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc = run_epoch(model, test_loader, criterion, device)

    elapsed = time.time() - started

    print(f"\nTrained in {elapsed / 60:.1f} min")
    print(f"Best validation accuracy: {state['best_val_acc']:.1%}")
    print(f"Held-out test accuracy:   {test_acc:.1%}")

    metrics = {
        "val_accuracy": round(state["best_val_acc"], 4),
        "test_accuracy": round(test_acc, 4),
        "test_loss": round(test_loss, 4),
        "epochs": args.epochs + (0 if args.no_finetune else args.finetune_epochs),
        "seed": args.seed,
    }

    save_checkpoint(model, classes, args.output, metrics)

    # a run sent elsewhere by --output keeps its history with it. Writing
    # to the one path regardless means a two-epoch trial overwrites the
    # record of the run the saved checkpoint actually came from, and the
    # curve and the weights then describe different models
    history_path = (
        HISTORY_PATH if args.output == CHECKPOINT_PATH
        else args.output.with_name(f"{args.output.stem}_history.json")
    )

    history_path.write_text(
        json.dumps({"metrics": metrics, "history": state["history"]}, indent=2)
    )

    print(f"\nSaved model to {args.output}")
    print(f"Saved history to {history_path}")


if __name__ == "__main__":
    main()
