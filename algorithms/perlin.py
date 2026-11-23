"""Perlin noise (Ken Perlin's 2002 "improved noise"), 2D, from scratch, standard library only.

    python perlin.py                      # writes output/perlin.png
    python perlin.py --seed 7 --octaves 1 --scale 32 --out output/single.png
"""
import argparse
import os
import random
import struct
import zlib


def make_permutation(seed):
    """Shuffle 0..255 with the seed, then repeat it so indices up to 511 need no wrap."""
    p = list(range(256))
    random.Random(seed).shuffle(p)
    return p + p


def fade(t):
    """6t^5 - 15t^4 + 10t^3: flat (zero 1st and 2nd derivative) at t = 0 and 1."""
    return t * t * t * (t * (t * 6 - 15) + 10)


def lerp(a, b, t):
    return a + t * (b - a)


def grad(h, x, y):
    """Dot product of (x, y) with one of 8 gradient directions chosen by the hash."""
    return (x + y, -x + y, x - y, -x - y, x, -x, y, -y)[h & 7]


def noise(perm, x, y):
    """One octave of Perlin noise at (x, y). Roughly in [-1, 1], exactly 0 at integer points."""
    # 1. Find the lattice cell containing the point.
    xi, yi = int(x // 1) & 255, int(y // 1) & 255
    # 2. Position inside that cell, in [0, 1).
    xf, yf = x - x // 1, y - y // 1
    # 3. Hash the four corners to pick their gradients, and dot each gradient
    #    with the vector from that corner to the point.
    a, b = perm[xi] + yi, perm[xi + 1] + yi
    g00 = grad(perm[a], xf, yf)
    g10 = grad(perm[b], xf - 1, yf)
    g01 = grad(perm[a + 1], xf, yf - 1)
    g11 = grad(perm[b + 1], xf - 1, yf - 1)
    # 4. Blend the four values, using the faded position as weights.
    u, v = fade(xf), fade(yf)
    return lerp(lerp(g00, g10, u), lerp(g01, g11, u), v)


def fbm(perm, x, y, octaves=5, lacunarity=2.0, gain=0.5):
    """Fractal Brownian motion: sum octaves, each at higher frequency and lower amplitude."""
    total, amp, freq, norm = 0.0, 1.0, 1.0, 0.0
    for _ in range(octaves):
        total += noise(perm, x * freq, y * freq) * amp
        norm += amp
        amp *= gain
        freq *= lacunarity
    return total / norm


def write_png(path, pixels, width, height):
    """Write 8-bit greyscale rows (list of bytes) as a PNG."""
    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))
    raw = b"".join(b"\x00" + row for row in pixels)  # filter type 0 per row
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 9)))
        f.write(chunk(b"IEND", b""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=256, help="image width and height in pixels")
    ap.add_argument("--scale", type=float, default=64, help="pixels per lattice cell (bigger = smoother)")
    ap.add_argument("--octaves", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="output/perlin.png")
    args = ap.parse_args()

    perm = make_permutation(args.seed)
    rows, lo, hi = [], 1.0, -1.0
    for py in range(args.size):
        values = [fbm(perm, px / args.scale, py / args.scale, args.octaves) for px in range(args.size)]
        lo, hi = min(lo, *values), max(hi, *values)
        # Map [-1, 1] to grey 0..255 (black = low, white = high).
        rows.append(bytes(max(0, min(255, int((v + 1) * 127.5))) for v in values))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_png(args.out, rows, args.size, args.size)
    print(f"wrote {args.out} ({args.size}x{args.size}), noise range {lo:.3f} .. {hi:.3f}")


if __name__ == "__main__":
    main()
