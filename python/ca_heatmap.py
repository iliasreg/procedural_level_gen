#!/usr/bin/env python3
"""2D heatmap of a 3D cellular-automata (CA) terrain.

Prototype for the terrain-algorithm comparison (see docs/cellular-automata-terrain.md).

The CA works on a 3D voxel volume (True = solid). Its "heatmap" is the height of the
top-most solid voxel in each (x, z) column, i.e. the same kind of value a noise generator
returns from surface_height(x, z). A horizontal slice of the volume is drawn next to it to
show the caves and overhangs a plain heightmap cannot represent.

Usage:
    python ca_heatmap.py                       # writes output/heatmaps.png
    python ca_heatmap.py --size 192 --seed 7 --show
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy import ndimage


def neighbour_count(solid: np.ndarray) -> np.ndarray:
    """Number of solid voxels among the 26 neighbours (3x3x3 cube minus the centre).

    Outside the volume: below is solid (bedrock), above is air, the sides repeat the edge.
    """
    padded = np.pad(solid, 1, mode="edge")
    padded[:, 0, :] = 1    # axis 1 is height (y)
    padded[:, -1, :] = 0
    n_x, n_y, n_z = solid.shape
    total = np.zeros(solid.shape, dtype=np.uint8)
    for dx in (0, 1, 2):
        for dy in (0, 1, 2):
            for dz in (0, 1, 2):
                if (dx, dy, dz) != (1, 1, 1):
                    total += padded[dx:dx + n_x, dy:dy + n_y, dz:dz + n_z]
    return total


def ca_terrain(size: int, height: int, seed: int, p_bottom: float = 0.55, p_top: float = 0.42,
               birth: int = 14, survive: int = 13, iterations: int = 8) -> np.ndarray:
    """Generate a boolean voxel volume of shape (size, height, size); True = solid.

    1. Random fill whose density falls linearly with height (dense ground, sparse sky),
       so the smoothing step settles into a ground surface instead of uniform cheese.
    2. Synchronous 3D smoothing: air -> solid if >= `birth` solid neighbours,
       solid stays solid if >= `survive`.
    3. Keep only the solid mass connected to the bottom layer (removes floating specks).
    """
    rng = np.random.default_rng(seed)
    p = np.linspace(p_bottom, p_top, height)[None, :, None]
    solid = (rng.random((size, height, size)) < p).astype(np.uint8)
    solid[:, 0, :] = 1                                   # bedrock
    for _ in range(iterations):
        k = neighbour_count(solid)
        solid = np.where(solid == 1, k >= survive, k >= birth).astype(np.uint8)
        solid[:, 0, :] = 1
    labels, _ = ndimage.label(solid)                     # 6-connected components
    grounded = np.unique(labels[:, 0, :])
    return np.isin(labels, grounded[grounded > 0])


def top_height(volume: np.ndarray) -> np.ndarray:
    """Height of the top-most solid voxel per (x, z) column, shape (x, z)."""
    h = volume.shape[1]
    steps_from_top = np.argmax(volume[:, ::-1, :], axis=1)   # steps down from the top to the first solid
    return (h - 1 - steps_from_top).astype(float)


def roughness(hm: np.ndarray) -> tuple[float, float]:
    """(mean, max) absolute height step between neighbouring columns, in blocks."""
    steps = np.concatenate([np.abs(np.diff(hm, axis=0)).ravel(), np.abs(np.diff(hm, axis=1)).ravel()])
    return float(steps.mean()), float(steps.max())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=128, help="map width/depth in blocks")
    ap.add_argument("--height", type=int, default=48, help="volume height in blocks")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", type=Path, default=Path("output/heatmaps.png"))
    ap.add_argument("--show", action="store_true", help="open an interactive window")
    args = ap.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = time.perf_counter()
    volume = ca_terrain(args.size, args.height, args.seed)
    elapsed = time.perf_counter() - t
    hm = top_height(volume)
    mean_step, max_step = roughness(hm)

    print(f"size={args.size}x{args.size}, height={args.height}, seed={args.seed}")
    print(f"generation time : {elapsed:.3f} s")
    print(f"height          : mean {hm.mean():.1f}, std {hm.std():.1f}, min {hm.min():.0f}, max {hm.max():.0f}")
    print(f"column step     : avg {mean_step:.2f}, max {max_step:.0f} blocks")
    print(f"solid ratio     : {volume.mean():.2f}")

    mid = args.height // 3
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    im = axes[0].imshow(hm.T, cmap="terrain", vmin=0, vmax=args.height - 1, origin="upper")
    axes[0].set_title("3D cellular automaton\ntop height per column")
    axes[0].axis("off")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.02, label="height (blocks)")
    axes[1].imshow(volume[:, mid, :].T, cmap="gray_r", origin="upper")
    axes[1].set_title(f"Slice at y={mid}\nblack = solid, white = air (caves)")
    axes[1].axis("off")
    fig.suptitle(f"Cellular-automata terrain, seed {args.seed}")

    if args.show:
        plt.show()
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=110, bbox_inches="tight")
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
