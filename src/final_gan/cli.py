from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import torch

from final_gan.config import deep_update, load_config
from final_gan.data import build_dataloader
from final_gan.metrics import evaluate_generator
from final_gan.training import train
from final_gan.utils import get_device
from final_gan.visualize import load_generator, save_interpolations, save_samples


def cmd_train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.data_root:
        config["data"]["root"] = args.data_root
    if args.output_dir:
        config["paths"]["output_dir"] = args.output_dir
    train(config, resume_from=args.resume)


def cmd_generate(args: argparse.Namespace) -> None:
    save_samples(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        num_images=args.num_images,
        nrow=args.nrow,
        device=args.device,
    )
    print(f"Saved samples to {args.output}")


def cmd_interpolate(args: argparse.Namespace) -> None:
    save_interpolations(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        pairs=args.pairs,
        steps=args.steps,
        space=args.space,
        device=args.device,
    )
    print(f"Saved interpolations to {args.output_dir}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    generator, config = load_generator(args.checkpoint, device)
    if args.config:
        override_config = load_config(args.config)
        config = deep_update(config, {"eval": override_config.get("eval", {})})
    if args.data_root:
        config["data"]["root"] = args.data_root
    if args.num_images:
        config["eval"]["num_images"] = args.num_images
    dataloader = build_dataloader(
        {**config, "train": {**config["train"], "batch_size": config["eval"].get("batch_size", 64)}},
        shuffle=False,
    )
    metrics = evaluate_generator(
        generator,
        dataloader,
        model_name=config["model"].get("name", "dcgan"),
        z_dim=int(config["model"].get("z_dim", 128)),
        device=device,
        num_images=int(config["eval"].get("num_images", 2048)),
        batch_size=int(config["eval"].get("batch_size", 64)),
        diversity_config=config["eval"].get("diversity", {}),
    )
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

    output_path = args.output
    if output_path is None:
        output_dir = Path(config.get("paths", {}).get("output_dir", args.checkpoint.parent.parent))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / "evaluations" / f"{args.checkpoint.stem}_n{config['eval'].get('num_images', 2048)}_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(args.checkpoint),
        "data_root": str(config["data"]["root"]),
        "model": config["model"].get("name", "dcgan"),
        "num_images": int(config["eval"].get("num_images", 2048)),
        "batch_size": int(config["eval"].get("batch_size", 64)),
        "diversity": config["eval"].get("diversity", {}),
        "metrics": metrics,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved evaluation to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LFW face generation with DCGAN and StyleGAN-lite.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model from a YAML config.")
    train_parser.add_argument("--config", required=True, type=Path)
    train_parser.add_argument("--data-root", default=None)
    train_parser.add_argument("--output-dir", default=None)
    train_parser.add_argument("--resume", default=None, type=Path)
    train_parser.set_defaults(func=cmd_train)

    gen_parser = subparsers.add_parser("generate", help="Generate a sample grid from a checkpoint.")
    gen_parser.add_argument("--checkpoint", required=True, type=Path)
    gen_parser.add_argument("--output", required=True, type=Path)
    gen_parser.add_argument("--num-images", type=int, default=64)
    gen_parser.add_argument("--nrow", type=int, default=8)
    gen_parser.add_argument("--device", default="auto")
    gen_parser.set_defaults(func=cmd_generate)

    interp_parser = subparsers.add_parser("interpolate", help="Generate latent interpolation grids.")
    interp_parser.add_argument("--checkpoint", required=True, type=Path)
    interp_parser.add_argument("--output-dir", required=True, type=Path)
    interp_parser.add_argument("--pairs", type=int, default=2)
    interp_parser.add_argument("--steps", type=int, default=11)
    interp_parser.add_argument("--space", choices=["z", "w"], default="z")
    interp_parser.add_argument("--device", default="auto")
    interp_parser.set_defaults(func=cmd_interpolate)

    eval_parser = subparsers.add_parser("evaluate", help="Compute FID and Inception Score.")
    eval_parser.add_argument("--checkpoint", required=True, type=Path)
    eval_parser.add_argument("--config", type=Path, default=None, help="Optional evaluation override YAML.")
    eval_parser.add_argument("--data-root", default=None)
    eval_parser.add_argument("--num-images", type=int, default=None)
    eval_parser.add_argument("--device", default="auto")
    eval_parser.add_argument("--output", type=Path, default=None)
    eval_parser.set_defaults(func=cmd_evaluate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
