#!/usr/bin/env python3
"""Side-by-side 2D heatmaps: Perlin fBm, Simplex fBm and 3D cellular-automata terrain.

Prototype for the terrain-algorithm comparison (see docs/cellular-automata-terrain.md).

  * Perlin and Simplex are implemented from scratch (2D, vectorised with numpy),
    summed with fBm octaves like perlin.gd in the Godot project.
  * The cellular automaton (CA) works on a 3D voxel volume. Its "heatmap" is the
    height of the top-most solid voxel in each (x, z) column, i.e. the same kind of
    value the noise generators return from surface_height(x, z).

Usage:
    python heatmap_compare.py                       # writes output/heatmaps.png
    python heatmap_compare.py --size 192 --seed 7 --show
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

# --------------------------------------------------------------------------- noise

# 8 gradient directions (same idea as grad() in perlin.gd)
GRADS = np.array([(1, 1), (-1, 1), (1, -1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)], dtype=float)


def make_perm(seed: int) -> np.ndarray:
    """Seeded shuffled 0..255, stored twice so lookups never need to wrap."""
    p = np.random.default_rng(seed).permutation(256)
    return np.concatenate([p, p])


def fade(t: np.ndarray) -> np.ndarray:
    """Quintic fade 6t^5 - 15t^4 + 10t^3."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def perlin2d(x: np.ndarray, y: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Ken Perlin's improved noise in 2D, roughly in [-1, 1]."""
    xf, yf = np.floor(x), np.floor(y)
    xi, yi = xf.astype(int) & 255, yf.astype(int) & 255
    x, y = x - xf, y - yf
    u, v = fade(x), fade(y)
    a = perm[xi] + yi
    b = perm[xi + 1] + yi

    def corner(h, dx, dy):
        g = GRADS[perm[h] & 7]
        return g[..., 0] * dx + g[..., 1] * dy

    g00 = corner(a, x, y)
    g10 = corner(b, x - 1.0, y)
    g01 = corner(a + 1, x, y - 1.0)
    g11 = corner(b + 1, x - 1.0, y - 1.0)
    return (g00 + (g10 - g00) * u) * (1 - v) + (g01 + (g11 - g01) * u) * v


F2 = 0.5 * (np.sqrt(3.0) - 1.0)   # skew into the simplex (triangle) grid
G2 = (3.0 - np.sqrt(3.0)) / 6.0   # unskew back


def simplex2d(x: np.ndarray, y: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """2D simplex noise (Gustavson formulation): 3 triangle corners, radial falloff."""
    s = (x + y) * F2
    i, j = np.floor(x + s), np.floor(y + s)
    t = (i + j) * G2
    x0, y0 = x - (i - t), y - (j - t)
    i1 = (x0 > y0).astype(int)          # which of the two triangles of the cell we are in
    j1 = 1 - i1
    x1, y1 = x0 - i1 + G2, y0 - j1 + G2
    x2, y2 = x0 - 1.0 + 2.0 * G2, y0 - 1.0 + 2.0 * G2
    ii, jj = i.astype(int) & 255, j.astype(int) & 255

    def corner(dx, dy, di, dj):
        falloff = np.maximum(0.5 - dx * dx - dy * dy, 0.0)
        g = GRADS[perm[ii + di + perm[jj + dj]] & 7]
        return falloff**4 * (g[..., 0] * dx + g[..., 1] * dy)

    return 70.0 * (corner(x0, y0, 0, 0) + corner(x1, y1, i1, j1) + corner(x2, y2, 1, 1))


def fbm(fn, size: int, perm: np.ndarray, frequency: float, octaves: int) -> np.ndarray:
    """Sum octaves (frequency x2, amplitude x0.5), normalised by total amplitude."""
    ys, xs = np.mgrid[0:size, 0:size].astype(float)
    total, amp, norm = np.zeros((size, size)), 1.0, 0.0
    for _ in range(octaves):
        total += fn(xs * frequency, ys * frequency, perm) * amp
        norm += amp
        amp *= 0.5
        frequency *= 2.0
    return total / norm


# ------------------------------------------------------------------------------ CA

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
    flipped_first = np.argmax(volume[:, ::-1, :], axis=1)   # steps down from the top to first solid
    return (h - 1 - flipped_first).astype(float)


# --------------------------------------------------------------------------- report

def to_blocks(field: np.ndarray, height: int) -> np.ndarray:
    """Min-max rescale a noise field to integer block heights 0..height-1."""
    f = (field - field.min()) / (field.max() - field.min())
    return np.rint(f * (height - 1))


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

    perm = make_perm(args.seed)
    timings: dict[str, float] = {}

    t = time.perf_counter()
    perlin = to_blocks(fbm(perlin2d, args.size, perm, 0.02, 5), args.height)
    timings["Perlin fBm"] = time.perf_counter() - t

    t = time.perf_counter()
    simplex = to_blocks(fbm(simplex2d, args.size, perm, 0.02, 5), args.height)
    timings["Simplex fBm"] = time.perf_counter() - t

    t = time.perf_counter()
    volume = ca_terrain(args.size, args.height, args.seed)
    ca = top_height(volume)
    timings["3D cellular automaton"] = time.perf_counter() - t

    maps = {"Perlin fBm": perlin, "Simplex fBm": simplex, "3D cellular automaton": ca}

    print(f"size={args.size}x{args.size}, height={args.height}, seed={args.seed}")
    print(f"{'algorithm':<24}{'time (s)':>9}{'mean':>8}{'std':>7}{'avg step':>10}{'max step':>10}")
    for name, hm in maps.items():
        mean_step, max_step = roughness(hm)
        print(f"{name:<24}{timings[name]:>9.3f}{hm.mean():>8.1f}{hm.std():>7.1f}{mean_step:>10.2f}{max_step:>10.0f}")
    print(f"CA solid ratio: {volume.mean():.2f}")

    mid = args.height // 3
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.4))
    for ax, (name, hm) in zip(axes, maps.items()):
        im = ax.imshow(hm.T, cmap="terrain", vmin=0, vmax=args.height - 1, origin="upper")
        ax.set_title(f"{name}\ntop height per column")
        ax.axis("off")
    fig.colorbar(im, ax=axes[:3], fraction=0.015, pad=0.01, label="height (blocks)")
    axes[3].imshow(volume[:, mid, :].T, cmap="gray_r", origin="upper")
    axes[3].set_title(f"CA slice at y={mid}\nblack = solid, white = air (caves)")
    axes[3].axis("off")
    fig.suptitle(f"Terrain heatmaps, seed {args.seed}")

    if args.show:
        plt.show()
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=110, bbox_inches="tight")
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
