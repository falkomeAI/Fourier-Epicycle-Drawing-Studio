"""
AI-based path extraction for complex drawings.

Pipeline:
  1. Optional background removal
  2. Neural LineartDetector / LineartAnimeDetector
  3. Sample ink pixels from the line map
  4. Build a short closed tour so Fourier gets
     one continuous path instead of raw CV contours

Requires (optional):
  pip install controlnet-aux
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from scipy.spatial import KDTree


# Lazy-loaded detectors (heavy torch models)
_lineart_detector = None
_lineart_anime_detector = None


def _get_lineart_detector(anime: bool = False):
    """Load lineart annotator once."""
    global _lineart_detector, _lineart_anime_detector
    try:
        from controlnet_aux import LineartAnimeDetector, LineartDetector
    except ImportError as exc:
        raise ImportError(
            "AI lineart needs controlnet-aux. Run:\n"
            "  pip install controlnet-aux\n"
            "First use downloads weights (~200MB)."
        ) from exc

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if anime:
        if _lineart_anime_detector is None:
            print("  Loading LineartAnimeDetector...")
            det = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
            _lineart_anime_detector = det.to(device)
        return _lineart_anime_detector

    if _lineart_detector is None:
        print("  Loading LineartDetector...")
        det = LineartDetector.from_pretrained("lllyasviel/Annotators")
        _lineart_detector = det.to(device)
    return _lineart_detector


def _maybe_rembg(bgr: np.ndarray) -> np.ndarray:
    """Isolate subject with rembg if available; otherwise return original."""
    try:
        from rembg import remove
    except ImportError:
        return bgr

    print("  rembg: isolating subject...")
    rgba = remove(bgr)
    if isinstance(rgba, bytes):
        arr = np.frombuffer(rgba, dtype=np.uint8)
        rgba = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if rgba is None:
        return bgr
    if rgba.ndim == 3 and rgba.shape[2] == 4:
        alpha = rgba[:, :, 3:4].astype(float) / 255.0
        rgb = rgba[:, :, :3].astype(float)
        # Composite onto white so lineart sees ink on clean paper
        out = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
        return out
    return bgr


def ai_lineart_map(
    bgr: np.ndarray,
    *,
    anime: bool = False,
    coarse: bool = False,
    detect_resolution: int = 768,
    use_rembg: bool = False,
) -> np.ndarray:
    """
    Run lineart model → grayscale ink map (dark lines).

    Returns HxW uint8 where low values = ink (same polarity as pencil on paper).
    """
    if use_rembg:
        bgr = _maybe_rembg(bgr)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    from PIL import Image

    pil = Image.fromarray(rgb)
    detector = _get_lineart_detector(anime=anime)

    if anime:
        result = detector(pil, detect_resolution=detect_resolution, image_resolution=detect_resolution)
    else:
        result = detector(
            pil,
            coarse=coarse,
            detect_resolution=detect_resolution,
            image_resolution=detect_resolution,
        )

    if hasattr(result, "convert"):
        arr = np.array(result.convert("L"))
    else:
        arr = np.asarray(result)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # LineartDetector returns white lines on black; invert → black ink on white
    # (controlnet_aux already inverts for LineartDetector; anime may differ)
    mean_val = float(arr.mean())
    if mean_val < 127:
        # white lines on dark bg → invert so ink is dark
        arr = 255 - arr

    return arr.astype(np.uint8)


def _ink_pixels(line_map: np.ndarray, threshold: int = 180, max_points: int = 8000) -> np.ndarray:
    """Sample dark pixels from AI line map as Nx2 (x, y) points."""
    # Adaptive threshold: keep darkest fraction as ink
    ink = (line_map < threshold).astype(np.uint8) * 255

    # Thin a bit so we don't sample fat strokes densely
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel, iterations=1)

    ys, xs = np.where(ink > 0)
    if len(xs) < 20:
        # Fallback: Canny on the line map
        edges = cv2.Canny(line_map, 50, 150)
        ys, xs = np.where(edges > 0)

    if len(xs) < 10:
        raise ValueError("AI lineart produced too few ink pixels — try anime=True or a clearer image")

    pts = np.column_stack([xs, ys]).astype(float)

    if len(pts) > max_points:
        # Uniform random subsample (preserve coverage)
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pts), size=max_points, replace=False)
        pts = pts[idx]

    return pts


def _christofides_tour(points: np.ndarray) -> np.ndarray:
    """
    Approximate shortest closed tour through points.

    Uses: MST → odd-degree matching (greedy) → Euler tour → shortcut → 2-opt.
    For large N, MST is built via Prim on a subsampled KD-tree nearest graph
    to stay memory-safe (full distance matrix would be O(n²)).
    """
    n = len(points)
    if n < 3:
        return points

    # Cap for dense graphs — still plenty for Fourier
    work = points
    if n > 2500:
        step = int(np.ceil(n / 2500))
        work = points[::step]
        n = len(work)

    # --- Prim MST via KDTree (k-NN graph, not full dense) ---
    k = min(32, n - 1)
    tree = KDTree(work)
    parent = [-1] * n
    key = np.full(n, np.inf)
    in_mst = np.zeros(n, dtype=bool)
    key[0] = 0.0

    for _ in range(n):
        u = int(np.argmin(np.where(in_mst, np.inf, key)))
        in_mst[u] = True
        dists, idxs = tree.query(work[u], k=k + 1)
        for d, v in zip(np.atleast_1d(dists), np.atleast_1d(idxs)):
            v = int(v)
            if v == u or in_mst[v]:
                continue
            if d < key[v]:
                key[v] = d
                parent[v] = u

    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    for v in range(1, n):
        u = parent[v]
        if u >= 0:
            adj[u].append(v)
            adj[v].append(u)

    # Odd-degree vertices
    odd = [i for i in range(n) if len(adj[i]) % 2 == 1]

    # Greedy minimum matching on odd vertices
    odd_set = set(odd)
    while len(odd_set) >= 2:
        a = odd_set.pop()
        others = list(odd_set)
        if not others:
            break
        dists = np.linalg.norm(work[others] - work[a], axis=1)
        j = int(np.argmin(dists))
        b = others[j]
        odd_set.remove(b)
        adj[a].append(b)
        adj[b].append(a)

    # Hierholzer Euler tour
    adj_multi = {i: list(neis) for i, neis in adj.items()}
    stack = [0]
    circuit: list[int] = []
    while stack:
        u = stack[-1]
        if adj_multi[u]:
            v = adj_multi[u].pop()
            adj_multi[v].remove(u)
            stack.append(v)
        else:
            circuit.append(stack.pop())
    circuit.reverse()

    # Shortcut: skip already-visited vertices → Hamiltonian
    seen = set()
    tour: list[int] = []
    for v in circuit:
        if v not in seen:
            seen.add(v)
            tour.append(v)
    if tour and tour[0] != tour[-1]:
        tour.append(tour[0])

    ordered = work[tour[:-1]]  # drop duplicate close

    # 2-opt polish
    ordered = _two_opt(ordered, max_iters=3)

    return ordered


def _two_opt(points: np.ndarray, max_iters: int = 3) -> np.ndarray:
    """Local 2-opt improvement on a closed tour."""
    pts = points.copy()
    n = len(pts)
    if n < 4:
        return pts
    # A quadratic pass adds little value to large raster point clouds and
    # can dominate extraction time. The MST + matching tour is already useful
    # at this scale; reserve 2-opt for smaller paths where it is affordable.
    if n > 1500:
        return pts

    improved = True
    it = 0
    while improved and it < max_iters:
        improved = False
        it += 1
        for i in range(n - 2):
            j_end = n - 1 if i == 0 else n
            for j in range(i + 2, j_end):
                a, b = pts[i], pts[(i + 1) % n]
                c, d = pts[j % n], pts[(j + 1) % n]
                before = float(np.hypot(*(a - b)) + np.hypot(*(c - d)))
                after = float(np.hypot(*(a - c)) + np.hypot(*(b - d)))
                if after + 1e-9 < before:
                    pts[i + 1 : j + 1] = pts[i + 1 : j + 1][::-1]
                    improved = True
        # 2-opt is O(n²) — one pass is enough for large point clouds
    return pts


def _resample_closed(points: np.ndarray, n: int) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        raise ValueError("contour too short")
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    diffs = np.diff(pts, axis=0)
    seg = np.hypot(diffs[:, 0], diffs[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total < 1e-9:
        raise ValueError("degenerate contour")
    targets = np.linspace(0.0, total, n, endpoint=False)
    x = np.interp(targets, cum, pts[:, 0])
    y = np.interp(targets, cum, pts[:, 1])
    return np.column_stack([x, y])


def extract_ai_path(
    image_path: str | Path,
    n_points: int = 4096,
    *,
    style: Literal["realistic", "anime"] = "realistic",
    coarse: bool = False,
    use_rembg: bool = False,
    detect_resolution: int = 768,
    ink_threshold: int = 180,
    debug_path: str | Path | None = None,
) -> np.ndarray:
    """
    Extract a single closed path using open-source AI lineart + TSP tour.

    This is the recommended method for complex photos / drawings when you
    want high internal detail (not just the outer silhouette).

    Args:
        image_path: Input image
        n_points: Output sample count for Fourier fitting
        style: "realistic" → LineartDetector, "anime" → LineartAnimeDetector
        coarse: Coarser realistic lineart (fewer tiny strokes)
        use_rembg: Pre-isolate subject with rembg
        detect_resolution: AI input size (higher = more detail, slower)
        ink_threshold: Pixels darker than this count as ink
        debug_path: Optional path to save AI line map overlay
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        # OpenCV builds often lack AVIF support; Pillow can decode it when
        # the corresponding plugin is installed. Convert RGB to OpenCV BGR.
        try:
            from PIL import Image

            rgb = np.asarray(Image.open(path).convert("RGB"))
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            raise ValueError(f"could not read image: {path}") from exc

    print(f"  AI lineart ({style}) @ {detect_resolution}px...")
    line_map = ai_lineart_map(
        img,
        anime=(style == "anime"),
        coarse=coarse,
        detect_resolution=detect_resolution,
        use_rembg=use_rembg,
    )

    # Match original aspect for coordinate space: resize line map to original
    h, w = img.shape[:2]
    line_map = cv2.resize(line_map, (w, h), interpolation=cv2.INTER_AREA)

    print("  Sampling ink + Christofides tour...")
    # The tour builder caps its working graph at 2500 points, so sampling
    # more cannot improve coverage and only adds work before downsampling.
    raw = _ink_pixels(line_map, threshold=ink_threshold, max_points=min(2500, max(n_points * 2, 2000)))
    ordered = _christofides_tour(raw)

    pts = _resample_closed(ordered, n_points)

    # Image → math coords
    pts[:, 1] = -pts[:, 1]
    pts -= pts.mean(axis=0)

    if debug_path is not None:
        dbg = img.copy()
        # Overlay line map in green
        ink = (line_map < ink_threshold).astype(np.uint8) * 255
        overlay = dbg.copy()
        overlay[ink > 0] = (0, 220, 80)
        blend = cv2.addWeighted(dbg, 0.45, overlay, 0.55, 0)
        # Draw tour polyline
        tour_vis = ordered.copy()
        tour_vis = tour_vis.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(blend, [tour_vis], False, (0, 140, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(debug_path), blend)
        # Also save pure line map
        map_path = Path(debug_path).with_name(Path(debug_path).stem + "_linemap.png")
        cv2.imwrite(str(map_path), line_map)
        print(f"  Debug → {debug_path}")
        print(f"  Line map → {map_path}")

    return pts
