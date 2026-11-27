"""Render the superradiant burst as a glowing cavity scene, driven frame by frame by the exact quantum solution.

Nothing here is hand-animated. For held-out cell H11 (5 atoms; results/quantum_states_H11.npz from
scripts/export_quantum_states.py):
  - each atom glows with its own exact excitation probability p_i(t) = (1 + <sz_i>) / 2,
  - the standing-wave cavity mode glows with the exact mean photon number <a+a>(t),
  - the beam leaving the output mirror carries the exact output flux kappa <a+a>(t), delayed by its travel time,
  - the inset is the exact <a+a>(t) curve.
The geometry (mirror shape, mode waist, atom spacing) is illustrative, not to scale.

    python run.py scripts/render_cavity.py

Writes assets/img/cavity.gif (README banner) and assets/img/cavity-peak.png (the frame at the burst peak).
"""
import sys
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw, ImageFont

SRC = "results/quantum_states_H11.npz"
W, H = 820, 312             # output size
SS = 2                      # supersampling
FPS, T_END, HOLD = 12, 15.0, 10

FIELD = np.array([0.32, 0.72, 1.00])
ATOM = np.array([1.00, 0.66, 0.24])
ATOM_DIM = np.array([0.42, 0.48, 0.60])
BEAM = np.array([0.40, 0.80, 1.00])


def font(size, mono=False):
    path = "/System/Library/Fonts/SFNSMono.ttf" if mono else "/System/Library/Fonts/HelveticaNeue.ttc"
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def build_scene(M, seed=7):
    w, h = W * SS, H * SS
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    y0 = 0.50 * h
    xL, xR = 0.10 * w, 0.68 * w
    u = (x - xL) / (xR - xL)
    inside = (u > 0) & (u < 1)
    w0 = 0.085 * h
    wz = w0 * np.sqrt(1 + ((u - 0.5) / 0.42) ** 2)
    q = 9
    mode = np.where(inside, np.cos(np.pi * q * u) ** 2 * np.exp(-2 * (y - y0) ** 2 / wz ** 2), 0.0)
    mode = mode * (0.35 + 0.65 * np.exp(-2 * (y - y0) ** 2 / (0.7 * wz) ** 2))

    # atoms sit on antinodes, nudged off-axis (the disorder is in their frequencies; this is just for the eye)
    rng = np.random.default_rng(seed)
    slots = [1, 2, 4, 6, 7][:M] if M <= 5 else list(range(1, M + 1))
    atoms = []
    for k in slots:
        ax = xL + (xR - xL) * (k / q)
        ay = y0 + rng.uniform(-0.45, 0.45) * w0
        r2 = (x - ax) ** 2 + (y - ay) ** 2
        core = np.exp(-r2 / (2 * (0.016 * h) ** 2))
        halo = np.exp(-r2 / (2 * (0.07 * h) ** 2))
        atoms.append((ax, ay, core, halo, r2))

    # output beam to the right of the output coupler
    xo = xR + 0.022 * w
    d = (x - xo) / (w - xo)
    beam_shape = np.where(d > 0, np.exp(-2 * (y - y0) ** 2 / (w0 * (1 + 0.6 * d)) ** 2) * np.exp(-1.4 * d), 0.0)

    # mirrors: concave faces toward the cavity, metal with a soft vertical gradient and a specular band
    c = 0.55 / h
    half = 0.25 * h
    thick = 0.020 * w
    sL = xL - 0.012 * w + c * (y - y0) ** 2
    sR = xR + 0.012 * w - c * (y - y0) ** 2
    inY = np.abs(y - y0) < half
    mL = inY & (x <= sL) & (x >= sL - thick)
    mR = inY & (x >= sR) & (x <= sR + thick * 0.8)
    tone = 0.55 + 0.35 * np.cos(np.pi * (y - y0) / (2 * half)) ** 2
    spec_band = np.exp(-((y - (y0 - 0.45 * half)) / (0.06 * h)) ** 2)
    metal = np.zeros((h, w, 3), np.float32)
    for m, tint in ((mL, np.array([0.80, 0.83, 0.88])), (mR, np.array([0.70, 0.80, 0.92]))):
        metal[m] = (tint * (tone[m] * 0.55 + 0.35 * spec_band[m])[:, None])
    face = np.zeros((h, w), np.float32)
    face[(inY & (np.abs(x - sL) < 2.2 * SS)) | (inY & (np.abs(x - sR) < 2.2 * SS))] = 1.0
    mirror_mask = (mL | mR).astype(np.float32)

    vign = 1 - 0.55 * (((x / w - 0.5) / 0.75) ** 2 + ((y / h - 0.5) / 0.9) ** 2)
    bg = np.clip(vign, 0.35, 1)[..., None] * np.array([0.018, 0.024, 0.040])
    # faint optical-table grid
    grid = ((np.mod(x, 40 * SS) < 1.0 * SS) | (np.mod(y, 40 * SS) < 1.0 * SS)).astype(np.float32) * 0.012
    bg = bg + grid[..., None] * np.clip(vign, 0, 1)[..., None]
    R = 0.026 * h
    return dict(R=R, xx=x, yy=y, mode=mode, atoms=atoms, beam=beam_shape, d=d, metal=metal, face=face, mirror=mirror_mask,
                bg=bg.astype(np.float32), xo=xo, y0=y0, w0=w0)


def frame(S, n_now, pe_now, n_hist, t_now, nmax, ts, ns):
    h, w = S["mode"].shape
    A = n_now / nmax
    E = np.zeros((h, w, 3), np.float32)
    E += (1.0 * A) * S["mode"][..., None] * FIELD
    for (ax, ay, core, halo, r2), p in zip(S["atoms"], pe_now):
        E += (1.8 * p * core + 0.7 * p * halo)[..., None] * ATOM
    # beam: brightness at distance d is the output flux emitted d/v earlier
    lag = S["d"] * 2.2
    bright = np.interp(np.clip(t_now - lag, 0, None), ts, ns) / nmax * (t_now - lag > 0)
    E += (0.95 * bright * S["beam"])[..., None] * BEAM
    E += (0.9 * A * S["face"])[..., None] * FIELD
    bloom = gaussian_filter(E, sigma=(8 * SS, 8 * SS, 0)) * 0.7 + gaussian_filter(E, sigma=(30 * SS, 30 * SS, 0)) * 0.45
    L = E + bloom
    img = 1 - np.exp(-1.35 * L)
    img = img * (1 - S["mirror"][..., None]) + S["metal"] + S["mirror"][..., None] * 0.18 * A * FIELD
    img = img + S["bg"] * (1 - img)
    # the atoms themselves: small shaded spheres, grey-blue in the ground state, amber when excited
    R = S["R"]
    for (ax, ay, core, halo, r2), p in zip(S["atoms"], pe_now):
        rr = np.sqrt(r2) / R
        alpha = np.clip((1.0 - rr) * R / 1.5, 0, 1)
        if not alpha.any():
            continue
        nz = np.sqrt(np.clip(1 - rr ** 2, 0, 1))
        yy, xx = S["yy"], S["xx"]
        nx, ny = (xx - ax) / R, (yy - ay) / R
        lam = np.clip(-0.45 * nx - 0.55 * ny + 0.70 * nz, 0, 1)
        spec = np.clip(-0.45 * nx - 0.55 * ny + 0.70 * nz, 0, 1) ** 24
        base = (1 - p) * np.array([0.50, 0.56, 0.68]) + p * np.array([1.0, 0.74, 0.36])
        col = (0.25 + 0.75 * lam)[..., None] * base + (0.85 * spec)[..., None] + (0.5 * p * nz)[..., None] * ATOM
        img = img * (1 - alpha[..., None]) + np.clip(col, 0, 1) * alpha[..., None]
    img = np.clip(img, 0, 1) ** (1 / 1.12)
    im = Image.fromarray((img * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS)
    return im


def hud(im, t_now, n_now, ts, ns, nmax, M, pe_now):
    d = ImageDraw.Draw(im, "RGBA")
    d.text((24, 16), "SUPERRADIANT BURST", font=font(15), fill=(237, 237, 237, 235))
    d.text((24, 37), f"{M} atoms in a lossy cavity, exact quantum solution", font=font(14), fill=(161, 161, 161, 230))
    # inset: exact <a+a>(t)
    x0, y0, iw, ih = W - 200, H - 84, 174, 52
    d.rectangle([x0 - 12, y0 - 30, x0 + iw + 12, y0 + ih + 14], fill=(10, 10, 10, 170), outline=(48, 48, 48, 255))
    d.text((x0, y0 - 24), "photons in the cavity", font=font(12), fill=(161, 161, 161, 255))
    d.text((x0 + iw, y0 - 24), f"{n_now:4.2f}", font=font(12, mono=True), fill=(80, 180, 255, 255), anchor="ra")
    sel = ts <= T_END
    px = x0 + ts[sel] / T_END * iw
    py = y0 + ih - ns[sel] / nmax * ih
    d.line(list(zip(px, py)), fill=(70, 70, 70, 255), width=1)
    k = ts <= t_now
    if k.sum() > 1:
        d.line(list(zip(px[k[sel]], py[k[sel]])), fill=(80, 180, 255, 255), width=2)
    cx, cy = x0 + t_now / T_END * iw, y0 + ih - n_now / nmax * ih
    d.ellipse([cx - 3.5, cy - 3.5, cx + 3.5, cy + 3.5], fill=(237, 237, 237, 255))
    d.text((26, H - 40), f"t = {t_now:5.2f}", font=font(13, mono=True), fill=(161, 161, 161, 255))
    # atom excitation bars
    bx = 26
    d.text((bx, H - 74), "atom excitation", font=font(12), fill=(161, 161, 161, 255))
    for i, p in enumerate(pe_now):
        X = bx + 112 + i * 16
        d.rectangle([X, H - 76, X + 10, H - 60], outline=(60, 60, 60, 255))
        d.rectangle([X, H - 60 - 16 * p, X + 10, H - 60], fill=(255, 168, 60, 255))
    return im


def main():
    z = np.load(SRC)
    ts, ns, pe = z["t"], z["n"], z["p_exc"]
    M = pe.shape[1]
    nmax = ns.max()
    S = build_scene(M)
    times = list(np.linspace(0, T_END, int(T_END * 4) + 1))
    ipk = int(np.argmax(ns))
    t_pk = float(ts[ipk])
    peak = frame(S, float(ns[ipk]), [float(v) for v in pe[ipk]], None, t_pk, nmax, ts, ns)
    hud(peak, t_pk, float(ns[ipk]), ts, ns, nmax, M, [float(v) for v in pe[ipk]]).save("assets/img/cavity-peak.png")
    if "--peak-only" in sys.argv:
        return
    frames = []
    for t_now in times:
        n_now = float(np.interp(t_now, ts, ns))
        p_now = [float(np.interp(t_now, ts, pe[:, i])) for i in range(M)]
        im = frame(S, n_now, p_now, None, t_now, nmax, ts, ns)
        frames.append(hud(im, t_now, n_now, ts, ns, nmax, M, p_now))

    # one shared palette keeps the gradients from flickering between frames
    sample = Image.new("RGB", (W, H * 4))
    for j, f in enumerate([frames[0], frames[len(frames) // 5], frames[ipk * len(frames) // len(ts) + 2], frames[-1]]):
        sample.paste(f, (0, H * j))
    pal = sample.quantize(colors=200, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.FLOYDSTEINBERG) for f in frames]
    q += [q[-1]] * HOLD
    q[0].save("assets/img/cavity.gif", save_all=True, append_images=q[1:], duration=int(1000 / FPS), loop=0,
              optimize=True)
    print(f"wrote assets/img/cavity.gif ({len(q)} frames) and cavity-peak.png (t = {t_pk:.2f})")


if __name__ == "__main__":
    main()
