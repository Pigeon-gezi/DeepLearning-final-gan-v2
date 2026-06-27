from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "report" / "assets"

LOGS = {
    "DCGAN": ROOT / "runs" / "dcgan_celeba" / "train_log.jsonl",
    "StyleGAN-lite": ROOT / "runs" / "stylegan_lite_celeba" / "train_log.jsonl",
    "StyleGAN-lite-v2": ROOT / "runs" / "stylegan_lite_v2_celeba" / "train_log.jsonl",
}

EVALS = {
    "DCGAN": ROOT / "runs" / "dcgan_celeba" / "evaluations" / "best_fid_n10000_20260627_195503.json",
    "StyleGAN-lite": ROOT
    / "runs"
    / "stylegan_lite_celeba"
    / "evaluations"
    / "best_fid_n10000_20260627_211631.json",
    "StyleGAN-lite-v2": ROOT
    / "runs"
    / "stylegan_lite_v2_celeba"
    / "evaluations"
    / "best_fid_n10000_manual.json",
}

COLORS = {
    "DCGAN": "#3568a8",
    "StyleGAN-lite": "#bf3f3f",
    "StyleGAN-lite-v2": "#2d8a5f",
}


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_eval(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_rows(rows: Iterable[dict], key: str) -> tuple[list[int], list[float]]:
    epochs: list[int] = []
    values: list[float] = []
    for row in rows:
        if key in row:
            epochs.append(int(row["epoch"]))
            values.append(float(row[key]))
    return epochs, values


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_fid_curve(logs: dict[str, list[dict]]) -> None:
    plt.figure(figsize=(7.2, 4.2))
    for name, rows in logs.items():
        epochs, values = metric_rows(rows, "fid")
        plt.plot(epochs, values, marker="o", linewidth=2, label=name, color=COLORS[name])
    plt.xlabel("Epoch")
    plt.ylabel("FID (lower is better)")
    plt.title("FID during training")
    plt.grid(True, alpha=0.25)
    plt.legend()
    save_plot(ASSETS / "fid_curve.png")


def plot_losses(logs: dict[str, list[dict]]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.0), sharex=False)
    for ax, (name, rows) in zip(axes, logs.items()):
        epochs = [int(row["epoch"]) for row in rows]
        loss_g = [float(row["loss_g"]) for row in rows]
        loss_d = [float(row["loss_d"]) for row in rows]
        ax.plot(epochs, loss_g, label="Generator", color="#7851a9", linewidth=1.8)
        ax.plot(epochs, loss_d, label="Discriminator", color="#d18f28", linewidth=1.8)
        ax.set_title(name)
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Epoch")
    fig.suptitle("Generator and discriminator losses", y=1.01)
    save_plot(ASSETS / "loss_curves.png")


def plot_diversity(rows: list[dict]) -> None:
    epochs, ms_ssim = metric_rows(rows, "diversity_ms_ssim")
    _, diversity = metric_rows(rows, "diversity_score")
    plt.figure(figsize=(7.2, 4.0))
    plt.plot(epochs, ms_ssim, marker="o", label="MS-SSIM (lower)", color="#bf3f3f", linewidth=2)
    plt.plot(epochs, diversity, marker="o", label="1 - MS-SSIM (higher)", color="#2d8a5f", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("StyleGAN-lite-v2 diversity metrics")
    plt.grid(True, alpha=0.25)
    plt.legend()
    save_plot(ASSETS / "v2_diversity_curve.png")


def plot_final_bars(evals: dict[str, dict]) -> None:
    names = list(evals)
    fid = [float(evals[name]["metrics"]["fid"]) for name in names]
    is_mean = [float(evals[name]["metrics"]["is_mean"]) for name in names]
    is_std = [float(evals[name]["metrics"]["is_std"]) for name in names]
    colors = [COLORS[name] for name in names]

    plt.figure(figsize=(6.4, 4.0))
    plt.bar(names, fid, color=colors)
    plt.ylabel("FID (lower is better)")
    plt.title("Final FID on 10,000 generated samples")
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(fid):
        plt.text(idx, value + max(fid) * 0.02, f"{value:.2f}", ha="center", fontsize=9)
    save_plot(ASSETS / "final_fid_bar.png")

    plt.figure(figsize=(6.4, 4.0))
    plt.bar(names, is_mean, yerr=is_std, capsize=4, color=colors)
    plt.ylabel("Inception Score")
    plt.title("Final Inception Score on 10,000 generated samples")
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(is_mean):
        plt.text(idx, value + 0.08, f"{value:.2f}", ha="center", fontsize=9)
    save_plot(ASSETS / "final_is_bar.png")


def plot_metrics_dashboard(logs: dict[str, list[dict]], evals: dict[str, dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.0))
    ax = axes[0, 0]
    for name, rows in logs.items():
        epochs, values = metric_rows(rows, "fid")
        ax.plot(epochs, values, marker="o", linewidth=1.8, label=name, color=COLORS[name])
    ax.set_title("FID during training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("FID")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    names = list(evals)
    fid = [float(evals[name]["metrics"]["fid"]) for name in names]
    ax.bar(names, fid, color=[COLORS[name] for name in names])
    ax.set_title("Final FID")
    ax.set_ylabel("FID")
    ax.tick_params(axis="x", labelrotation=15)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    rows = logs["StyleGAN-lite-v2"]
    epochs, values = metric_rows(rows, "diversity_score")
    ax.plot(epochs, values, marker="o", color=COLORS["StyleGAN-lite-v2"], linewidth=1.8)
    ax.set_title("StyleGAN-lite-v2 diversity")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("1 - MS-SSIM")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    rows = logs["StyleGAN-lite-v2"]
    epochs = [int(row["epoch"]) for row in rows]
    ax.plot(epochs, [float(row["loss_g"]) for row in rows], label="G", color="#7851a9")
    ax.plot(epochs, [float(row["loss_d"]) for row in rows], label="D", color="#d18f28")
    ax.set_title("StyleGAN-lite-v2 losses")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_plot(ASSETS / "metrics_dashboard.png")


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arial.ttf", "msyh.ttc", "simhei.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def add_title(image: Image.Image, title: str) -> Image.Image:
    font = load_font(34)
    pad = 18
    title_h = 56
    out = Image.new("RGB", (image.width, image.height + title_h), "white")
    out.paste(image.convert("RGB"), (0, title_h))
    draw = ImageDraw.Draw(out)
    draw.text((pad, 10), title, fill=(20, 20, 20), font=font)
    return out


def resize_width(image: Image.Image, width: int) -> Image.Image:
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def concat_vertical(items: list[Image.Image], gap: int = 18) -> Image.Image:
    width = max(item.width for item in items)
    height = sum(item.height for item in items) + gap * (len(items) - 1)
    out = Image.new("RGB", (width, height), "white")
    y = 0
    for item in items:
        out.paste(item.convert("RGB"), ((width - item.width) // 2, y))
        y += item.height + gap
    return out


def concat_horizontal(items: list[Image.Image], gap: int = 18) -> Image.Image:
    height = max(item.height for item in items)
    width = sum(item.width for item in items) + gap * (len(items) - 1)
    out = Image.new("RGB", (width, height), "white")
    x = 0
    for item in items:
        out.paste(item.convert("RGB"), (x, (height - item.height) // 2))
        x += item.width + gap
    return out


def make_image_composites() -> None:
    sample_paths = [
        ("DCGAN", ROOT / "runs" / "dcgan_celeba" / "best_samples.png"),
        ("StyleGAN-lite", ROOT / "runs" / "stylegan_lite_celeba" / "samples" / "epoch_0100.png"),
        ("StyleGAN-lite-v2", ROOT / "runs" / "stylegan_lite_v2_celeba" / "samples" / "epoch_0150.png"),
    ]
    sample_items = []
    for title, path in sample_paths:
        image = resize_width(Image.open(path), 900)
        sample_items.append(add_title(image, title))
    concat_vertical(sample_items).save(ASSETS / "sample_grid_comparison.png")

    interp_paths = [
        ("DCGAN z", ROOT / "runs" / "dcgan_celeba" / "interpolation_z_00.png"),
        ("StyleGAN-lite-v2 z", ROOT / "runs" / "stylegan_lite_v2_celeba" / "interpolations" / "interpolation_z_00.png"),
        ("StyleGAN-lite-v2 w", ROOT / "runs" / "stylegan_lite_v2_celeba" / "interpolations" / "interpolation_w_00.png"),
    ]
    interp_items = []
    for title, path in interp_paths:
        image = resize_width(Image.open(path), 900)
        interp_items.append(add_title(image, title))
    concat_vertical(interp_items).save(ASSETS / "interpolation_comparison.png")

    # Copy key single images into report/assets with stable names.
    copies = {
        "dcgan_samples.png": ROOT / "runs" / "dcgan_celeba" / "best_samples.png",
        "stylegan_lite_failure_epoch_0020.png": ROOT / "runs" / "stylegan_lite_celeba" / "samples" / "epoch_0020.png",
        "stylegan_lite_failure_epoch_0100.png": ROOT / "runs" / "stylegan_lite_celeba" / "samples" / "epoch_0100.png",
        "stylegan_lite_v2_samples.png": ROOT / "runs" / "stylegan_lite_v2_celeba" / "samples" / "epoch_0150.png",
        "dcgan_interp_z.png": ROOT / "runs" / "dcgan_celeba" / "interpolation_z_00.png",
        "stylegan_v2_interp_z.png": ROOT
        / "runs"
        / "stylegan_lite_v2_celeba"
        / "interpolations"
        / "interpolation_z_00.png",
        "stylegan_v2_interp_w.png": ROOT
        / "runs"
        / "stylegan_lite_v2_celeba"
        / "interpolations"
        / "interpolation_w_00.png",
    }
    for target, source in copies.items():
        Image.open(source).save(ASSETS / target)


def draw_flow(path: Path, boxes: list[tuple[str, float, float, float, float]], arrows: list[tuple[int, int]], title: str) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title(title, fontsize=15, pad=12)
    for label, x, y, w, h in boxes:
        rect = Rectangle((x, y), w, h, linewidth=1.5, edgecolor="#2d4059", facecolor="#edf2f7")
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10, wrap=True)
    for start, end in arrows:
        _, x1, y1, w1, h1 = boxes[start]
        _, x2, y2, _, h2 = boxes[end]
        arrow = FancyArrowPatch(
            (x1 + w1, y1 + h1 / 2),
            (x2, y2 + h2 / 2),
            arrowstyle="->",
            mutation_scale=14,
            linewidth=1.5,
            color="#333333",
        )
        ax.add_patch(arrow)
    save_plot(path)


def make_diagrams() -> None:
    draw_flow(
        ASSETS / "project_pipeline.png",
        [
            ("CelebA\naligned faces", 0.2, 1.5, 1.5, 0.8),
            ("Preprocess\n64x64, [-1,1]", 2.0, 1.5, 1.6, 0.8),
            ("Train models\nDCGAN\nStyleGAN-lite\nStyleGAN-lite-v2", 3.75, 1.35, 2.65, 1.1),
            ("Samples\nand interpolation", 6.85, 2.3, 1.6, 0.8),
            ("FID / IS\nMS-SSIM", 6.85, 0.7, 1.6, 0.8),
            ("Report\nanalysis", 8.65, 1.5, 1.2, 0.8),
        ],
        [(0, 1), (1, 2), (2, 3), (2, 4), (3, 5), (4, 5)],
        "Project pipeline",
    )
    draw_flow(
        ASSETS / "gan_training_diagram.png",
        [
            ("Latent z", 0.3, 2.3, 1.2, 0.7),
            ("Generator G", 2.0, 2.3, 1.5, 0.7),
            ("Fake image", 4.0, 2.3, 1.4, 0.7),
            ("Real image", 4.0, 0.7, 1.4, 0.7),
            ("Discriminator D", 6.0, 1.5, 1.7, 0.8),
            ("Adversarial\nlosses", 8.2, 1.5, 1.4, 0.8),
        ],
        [(0, 1), (1, 2), (2, 4), (3, 4), (4, 5)],
        "GAN adversarial training",
    )
    draw_flow(
        ASSETS / "dcgan_architecture.png",
        [
            ("z\n128", 0.2, 1.5, 1.0, 0.7),
            ("Linear\n4x4x512", 1.6, 1.5, 1.4, 0.7),
            ("Deconv\n8x8x256", 3.3, 1.5, 1.4, 0.7),
            ("Deconv\n16x16x128", 5.0, 1.5, 1.4, 0.7),
            ("Deconv\n32x32x64", 6.7, 1.5, 1.4, 0.7),
            ("Tanh\n64x64x3", 8.4, 1.5, 1.2, 0.7),
        ],
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        "DCGAN generator",
    )
    draw_flow(
        ASSETS / "stylegan_lite_architecture.png",
        [
            ("z", 0.2, 2.3, 0.8, 0.6),
            ("Mapping\nMLP", 1.3, 2.3, 1.2, 0.6),
            ("w", 2.9, 2.3, 0.8, 0.6),
            ("Learned\nconstant", 1.3, 0.9, 1.2, 0.6),
            ("StyledConv\nAdaIN + noise", 4.0, 1.6, 1.8, 0.8),
            ("Upsample\nblocks", 6.2, 1.6, 1.5, 0.8),
            ("ToRGB\n64x64", 8.1, 1.6, 1.2, 0.8),
        ],
        [(0, 1), (1, 2), (2, 4), (3, 4), (4, 5), (5, 6)],
        "StyleGAN-lite generator",
    )
    draw_flow(
        ASSETS / "stylegan_lite_v2_architecture.png",
        [
            ("z", 0.2, 2.3, 0.8, 0.6),
            ("Mapping\n4-layer MLP", 1.3, 2.3, 1.3, 0.6),
            ("w", 3.0, 2.3, 0.8, 0.6),
            ("Learned\n4x4 constant", 1.3, 0.9, 1.3, 0.6),
            ("2x StyledConv\nper resolution", 4.0, 1.6, 1.8, 0.8),
            ("Noise + AdaIN\nstyle control", 6.2, 1.6, 1.7, 0.8),
            ("ToRGB\n64x64", 8.3, 1.6, 1.2, 0.8),
        ],
        [(0, 1), (1, 2), (2, 4), (3, 4), (4, 5), (5, 6)],
        "StyleGAN-lite-v2 generator",
    )
    draw_flow(
        ASSETS / "evaluation_pipeline.png",
        [
            ("Real images", 0.2, 2.3, 1.2, 0.7),
            ("Inception\nfeatures", 1.8, 2.3, 1.4, 0.7),
            ("Cached real\nstatistics", 3.6, 2.3, 1.4, 0.7),
            ("Generator", 0.2, 0.8, 1.2, 0.7),
            ("Fake images", 1.8, 0.8, 1.4, 0.7),
            ("FID / IS /\nMS-SSIM", 5.6, 1.5, 1.6, 0.8),
            ("JSON result\nand report", 7.8, 1.5, 1.5, 0.8),
        ],
        [(0, 1), (1, 2), (3, 4), (2, 5), (4, 5), (5, 6)],
        "Evaluation pipeline",
    )
    draw_flow(
        ASSETS / "improvement_path.png",
        [
            ("Observed issue\nmode collapse", 0.2, 1.5, 1.5, 0.8),
            ("Stronger synthesis\n2 StyledConv", 2.0, 2.3, 1.5, 0.8),
            ("TTUR + R1\nstable D", 4.0, 2.3, 1.5, 0.8),
            ("Minibatch stddev\npenalize repetition", 2.0, 0.7, 1.7, 0.8),
            ("EMA\nstable samples", 4.3, 0.7, 1.3, 0.8),
            ("StyleGAN-lite-v2\nFID 12.78", 6.5, 1.5, 1.6, 0.8),
        ],
        [(0, 1), (1, 2), (0, 3), (3, 4), (2, 5), (4, 5)],
        "Stabilization path",
    )


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    logs = {name: read_jsonl(path) for name, path in LOGS.items()}
    evals = {name: read_eval(path) for name, path in EVALS.items()}
    plot_fid_curve(logs)
    plot_losses(logs)
    plot_diversity(logs["StyleGAN-lite-v2"])
    plot_final_bars(evals)
    plot_metrics_dashboard(logs, evals)
    make_image_composites()
    make_diagrams()
    print(f"Saved report figures to {ASSETS}")


if __name__ == "__main__":
    main()
