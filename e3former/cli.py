"""Command-line interface for the four main protein benchmarks."""

import argparse
import json
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(prog="e3former")
    commands = p.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate"):
        cmd = commands.add_parser(name)
        cmd.add_argument(
            "--dataset", choices=("cath", "cath-af", "ppbs", "ppbs-af"), required=name == "train"
        )
        cmd.add_argument("--data-dir", type=Path)
        cmd.add_argument("--device", default="cuda")
        cmd.add_argument("--batch-size", type=int, default=8)
        cmd.add_argument("--workers", type=int, default=0)
        cmd.add_argument("--seed", type=int, default=52)
        cmd.add_argument("--limit-proteins", type=int)
        cmd.add_argument("--output", type=Path, required=True)
        cmd.add_argument("--verbose", action="store_true")
        if name == "train":
            cmd.add_argument("--features", choices=("seq", "angles", "backbone"), default="seq")
            cmd.add_argument("--encoder", default="e3former")
            cmd.add_argument("--epochs", type=int, default=50)
            cmd.add_argument("--patience", type=int, default=10)
            cmd.add_argument("--log-every", type=int, default=50)
            cmd.add_argument("--no-progress", action="store_true")
            cmd.add_argument("--learning-rate", type=float, default=0.001)
            cmd.add_argument("--base-radius", type=float)
            cmd.add_argument("--energy-radius", type=float)
            cmd.add_argument("--max-train-batches", type=int)
            cmd.add_argument("--max-val-batches", type=int)
            cmd.add_argument("--resume", type=Path)
        else:
            cmd.add_argument("--checkpoint", required=True)
            cmd.add_argument("--split", choices=("train", "validation", "test"), default="test")
            cmd.add_argument(
                "--subset", choices=("all", "70", "homology", "topology", "none"), default="all"
            )
            cmd.add_argument("--save-predictions", action="store_true")
    data = commands.add_parser("check-data")
    data.add_argument("--data-dir", type=Path)
    data.add_argument("--output", type=Path)
    commands.add_parser("checkpoints")
    return p


def main():
    args = parser().parse_args()
    if args.command == "checkpoints":
        from e3former.checkpoints import manifest

        print(json.dumps(manifest(), indent=2))
        return
    from e3former.data import audit, data_root, loader

    root = data_root(args.data_dir)
    if args.command == "check-data":
        result = audit(root)
        text = json.dumps(result, indent=2)
        print(text)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n")
        if any(
            v[split]["missing"]
            for v in result.values()
            for split in ("train", "validation", "test")
        ):
            raise SystemExit(1)
        if any(any(v["overlap"].values()) for v in result.values()):
            raise SystemExit("Split overlap detected; see the audit.")
        return

    import lightning as L
    import torch
    from loguru import logger
    from omegaconf import OmegaConf

    from e3former.checkpoints import checksum, load_checkpoint
    from e3former.config import ENCODERS, make_config
    from e3former.evaluate import evaluate
    from e3former.model import Model

    if args.batch_size < 1 or (args.limit_proteins is not None and args.limit_proteins < 1):
        raise SystemExit("Batch size and protein limit must be positive.")
    if not args.verbose:
        logger.remove()
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA unavailable. Check the CUDA installation or use --device cpu.")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    L.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    args.output.mkdir(parents=True, exist_ok=True)
    options = dict(
        root=root, batch_size=args.batch_size, workers=args.workers, limit=args.limit_proteins
    )
    if args.command == "evaluate":
        model, cfg, path = load_checkpoint(args.checkpoint, device)
        dataset = args.dataset or cfg.dataset_name
        if (dataset.startswith("cath")) != (cfg.task.task == "inverse_folding"):
            raise SystemExit("Checkpoint task does not match the evaluation dataset.")
        predictions = args.output / "predictions.csv" if args.save_predictions else None
        result = evaluate(
            model,
            loader(dataset=dataset, split=args.split, subset=args.subset, **options),
            device,
            predictions,
        )
        result.update(
            dataset=dataset,
            training_dataset=cfg.dataset_name,
            split=args.split,
            subset=args.subset,
            feature_set=cfg.feature_set,
            checkpoint=str(path),
            checkpoint_sha256=checksum(path),
            seed=args.seed,
            batch_size=args.batch_size,
            limited=args.limit_proteins is not None,
            deterministic_radius=cfg.encoder.get("deterministic_radius"),
            ssm_version=(
                cfg.encoder.get("ssm_version", 1) if cfg.encoder_name == "e3former" else None
            ),
            torch_version=torch.__version__,
        )
        (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        OmegaConf.save(cfg, args.output / "config.yaml")
        print(json.dumps(result, indent=2))
        return

    if args.encoder not in ENCODERS:
        raise SystemExit(f"Unknown encoder. Choose from {', '.join(ENCODERS)}.")
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger

    cfg = make_config(
        args.dataset, args.features, args.encoder, args.base_radius, args.energy_radius
    )
    cfg.optimiser.optimizer.lr = args.learning_rate
    cfg.seed = args.seed
    if args.resume:
        model, cfg, _ = load_checkpoint(str(args.resume), device)
        if cfg.dataset_name != args.dataset:
            raise SystemExit("Resume dataset must match checkpoint dataset.")
    else:
        model = Model(cfg).to(device)
    train = loader(dataset=args.dataset, split="train", shuffle=True, **options)
    val = loader(dataset=args.dataset, split="validation", **options)
    with torch.no_grad():
        model.eval()
        example = model.featurise(next(iter(val)).to(device))
        model(example)
        del example
    model.train()
    OmegaConf.save(cfg, args.output / "config.yaml")
    (args.output / "command.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    checkpoint = ModelCheckpoint(
        dirpath=args.output / "checkpoints",
        filename="best",
        monitor="val/loss",
        mode="min",
        save_last=True,
        save_top_k=1,
    )
    if args.epochs < 1 or args.patience < 0:
        raise SystemExit("Epochs must be positive; use --patience 0 to disable early stopping.")
    if args.log_every < 1:
        raise SystemExit("Log interval must be positive.")
    callbacks = [checkpoint]
    if args.patience:
        callbacks.append(EarlyStopping(monitor="val/loss", mode="min", patience=args.patience))
    trainer = L.Trainer(
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=[device.index or 0] if device.type == "cuda" else 1,
        max_epochs=args.epochs,
        callbacks=callbacks,
        logger=CSVLogger(str(args.output), name="logs"),
        limit_train_batches=args.max_train_batches if args.max_train_batches is not None else 1.0,
        limit_val_batches=args.max_val_batches if args.max_val_batches is not None else 1.0,
        num_sanity_val_steps=0,
        enable_model_summary=False,
        enable_progress_bar=not args.no_progress,
        log_every_n_steps=args.log_every,
    )
    trainer.fit(
        model,
        train_dataloaders=train,
        val_dataloaders=val,
        ckpt_path=str(args.resume) if args.resume else None,
    )
    if trainer.interrupted:
        raise SystemExit("Training interrupted. Resume from the latest completed checkpoint.")
    print(
        json.dumps(
            {
                "best_checkpoint": checkpoint.best_model_path,
                "last_checkpoint": checkpoint.last_model_path,
                "best_validation_loss": float(checkpoint.best_model_score),
                "completed_optimizer_steps": trainer.global_step,
                "epochs_completed": trainer.current_epoch,
            },
            indent=2,
        )
    )
