"""
Extract a closed (x, y) outline from an uploaded image.

Supports:
  - PNG with alpha (uses the alpha mask)
  - High-contrast / silhouette photos (Otsu threshold)
  - Natural photos via optional rembg background removal
  - Fallback GrabCut when rembg is not installed
  - LINE ART extraction with path ordering (for complex drawings)
"""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from typing import List, Tuple

import cv2
import numpy as np
from scipy.spatial import KDTree


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


def _otsu_threshold(gray: np.ndarray) -> int:
    """Otsu threshold with histogram and inclusive threshold rule."""
    hist = np.bincount(np.asarray(gray, dtype=np.uint8).ravel(), minlength=256)
    total = int(hist.sum())
    total_sum = sum(i * int(hist[i]) for i in range(256))
    sum_b = 0
    weight_b = 0
    best = -1.0
    threshold = 128
    for value in range(256):
        weight_b += int(hist[value])
        if weight_b == 0:
            continue
        weight_f = total - weight_b
        if weight_f == 0:
            break
        sum_b += value * int(hist[value])
        mean_b = sum_b / weight_b
        mean_f = (total_sum - sum_b) / weight_f
        between = weight_b * weight_f * (mean_b - mean_f) ** 2
        if between > best:
            best = between
            threshold = value
    return threshold


def _trace_all_moore(mask: np.ndarray, min_len: int = 25) -> list[np.ndarray]:
    """Port of the browser's Moore-neighbor ``traceAll`` contour walker."""
    h, w = mask.shape
    ring = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
    seen = np.zeros((h, w), dtype=bool)
    contours: list[np.ndarray] = []

    def at(row: int, col: int) -> bool:
        return 0 <= row < h and 0 <= col < w and bool(mask[row, col])

    for row in range(h):
        for col in range(w):
            if not at(row, col) or seen[row, col] or at(row, col - 1):
                continue
            back = 6
            pr, pc = row, col
            path = [(col, row)]
            for _ in range(4 * w * h):
                moved = False
                for offset in range(1, 9):
                    direction = (back + offset) % 8
                    dr, dc = ring[direction]
                    nr, nc = pr + dr, pc + dc
                    if at(nr, nc):
                        back = (direction + 5) % 8
                        pr, pc = nr, nc
                        moved = True
                        break
                if not moved or (pr == row and pc == col):
                    break
                path.append((pc, pr))
            for x, y in path:
                seen[y, x] = True
            if len(path) >= min_len:
                contours.append(np.asarray(path, dtype=float))
    contours.sort(key=len, reverse=True)
    return contours


def _nearest_pair_indices(a: np.ndarray, b: np.ndarray, budget: int = 600) -> tuple[int, int]:
    """Match the browser's coarse nearest-pair search and local refinement."""
    stride_a = max(1, int(np.ceil(len(a) / budget)))
    stride_b = max(1, int(np.ceil(len(b) / budget)))
    best_i = best_j = 0
    best_d = float("inf")
    for i in range(0, len(a), stride_a):
        delta = b[::stride_b] - a[i]
        distances = np.einsum("ij,ij->i", delta, delta)
        j_local = int(np.argmin(distances))
        d = float(distances[j_local])
        if d < best_d:
            best_d, best_i, best_j = d, i, j_local * stride_b

    lo_a, hi_a = max(0, best_i - stride_a), min(len(a), best_i + stride_a + 1)
    lo_b, hi_b = max(0, best_j - stride_b), min(len(b), best_j + stride_b + 1)
    aa, bb = a[lo_a:hi_a], b[lo_b:hi_b]
    delta = aa[:, None, :] - bb[None, :, :]
    distances = np.einsum("ijk,ijk->ij", delta, delta)
    i_local, j_local = np.unravel_index(int(np.argmin(distances)), distances.shape)
    return lo_a + int(i_local), lo_b + int(j_local)


def _chain_contours_continuous(contours: list[np.ndarray], n_points: int, max_shapes: int) -> np.ndarray:
    """Insert each contour as an out-and-back detour using KDTree for fast spatial queries."""
    pieces = [np.asarray(p, dtype=float) for p in contours if len(p) > 2]
    pieces.sort(key=len, reverse=True)
    pieces = pieces[:max_shapes]
    if not pieces:
        raise ValueError("no usable contours found — try changing the invert setting")

    path = np.vstack([pieces[0], pieces[0][0]])
    remaining = pieces[1:]

    while remaining:
        path_stride = max(1, len(path) // 400)
        path_sub = path[::path_stride]
        tree = KDTree(path_sub)
        best_piece = -1
        best_distance = float("inf")
        best_sub_idx = 0
        best_j = 0
        for piece_index, loop in enumerate(remaining):
            stride = max(1, len(loop) // 30)
            sub_loop = loop[::stride]
            dists, sub_idxs = tree.query(sub_loop)
            min_loc = int(np.argmin(dists))
            d = float(dists[min_loc])
            if d < best_distance:
                best_distance = d
                best_piece = piece_index
                best_sub_idx = int(sub_idxs[min_loc])
                best_j = min_loc * stride
        loop = remaining.pop(best_piece)
        approx_i = best_sub_idx * path_stride
        lo_i = max(0, approx_i - path_stride * 2)
        hi_i = min(len(path), approx_i + path_stride * 2 + 1)
        diffs = path[lo_i:hi_i] - loop[best_j]
        best_i = lo_i + int(np.argmin(np.einsum("ij,ij->i", diffs, diffs)))
        branch = np.vstack([loop[best_j:], loop[:best_j], loop[best_j]])
        path = np.vstack([path[:best_i + 1], branch, path[best_i:]])

    sampled = _resample_closed(path, n_points)
    sampled -= sampled.mean(axis=0)
    radius = float(np.max(np.hypot(sampled[:, 0], sampled[:, 1])))
    if radius > 0:
        sampled *= 10.0 / radius
    return sampled


def path_from_mask_contours(
    mask: np.ndarray, n_points: int, min_len: int = 8, max_contours: int = 500,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Build one closed, resampled path from all significant mask contours."""
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    paths = [
        contour.reshape(-1, 2).astype(float)
        for contour in contours
        if len(contour) >= min_len
    ]
    paths.sort(key=len, reverse=True)
    paths = paths[:max_contours]
    if not paths:
        inv_binary = (binary == 0).astype(np.uint8)
        inv_contours, _ = cv2.findContours(inv_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        paths = [
            contour.reshape(-1, 2).astype(float)
            for contour in inv_contours
            if len(contour) >= min_len
        ]
        paths.sort(key=len, reverse=True)
        paths = paths[:max_contours]
    if not paths:
        raise ValueError("no usable contours found — try another extraction method or adjust image")
    points = _chain_contours_continuous(paths, n_points, len(paths))
    points[:, 1] *= -1
    return points, paths


def extract_boundary_contour_path(
    image: np.ndarray,
    n_points: int = 3000,
    invert: bool = False,
    line_art: bool = True,
    min_len: int = 8,
    max_shapes: int = 500,
    resize_max: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Extract path without resizing to preserve fine edges and details."""
    if image is None or image.size == 0:
        raise ValueError("image is empty")
    h, w = image.shape[:2]

    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        bgra = image.astype(np.float32)
        alpha = bgra[:, :, 3:4] / 255.0
        bgr = np.clip(bgra[:, :, :3] * alpha + 255.0 * (1.0 - alpha), 0, 255).astype(np.uint8)
    else:
        bgr = image[:, :, :3].copy()

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if int(gray.max()) - int(gray.min()) < 2:
        raise ValueError("that image is one flat colour — nothing to trace")
    threshold = _otsu_threshold(gray)
    mask = ((gray <= threshold) != bool(invert)).astype(np.uint8)
    contours = _trace_all_moore(mask, min_len=min_len)
    if not contours:
        cv_cnts, _ = cv2.findContours(
            (mask * 255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        contours = [c.squeeze(axis=1).astype(float) for c in cv_cnts if len(c) >= min_len]
    if not contours:
        cv_cnts, _ = cv2.findContours(
            ((1 - mask) * 255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        contours = [c.squeeze(axis=1).astype(float) for c in cv_cnts if len(c) >= min_len]
    if not contours:
        adapt_mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
        )
        cv_cnts, _ = cv2.findContours(adapt_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = [c.squeeze(axis=1).astype(float) for c in cv_cnts if len(c) >= min_len]
    if not contours:
        raise ValueError("Could not detect contours in image — try another extraction method or invert")
    if not line_art:
        contours = contours[:1]
    # Draw on copy_and_resize image, but take reference always from original unresized image
    scale = min(1.0, 1200.0 / max(h, w))
    disp_w = max(1, int(round(w * scale)))
    disp_h = max(1, int(round(h * scale)))
    contour_preview = cv2.resize(bgr, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
    for contour in contours:
        scaled_poly = np.rint(contour * scale).astype(np.int32).reshape(-1, 1, 2)
        if len(scaled_poly) > 1:
            cv2.polylines(contour_preview, [scaled_poly], isClosed=True, color=(0, 220, 120), thickness=2, lineType=cv2.LINE_AA)
    points = _chain_contours_continuous(contours, n_points, max_shapes)
    points[:, 1] *= -1
    return points, bgr, mask * 255, contour_preview, len(contours)


def _largest_external_contour(mask: np.ndarray, smooth: bool = True) -> np.ndarray:
    """Return Nx2 float contour (image coords, y down) of the largest blob.
    
    Args:
        mask: Binary mask (any non-zero value is foreground)
        smooth: If True, apply morphological smoothing to clean up jagged edges
    
    Returns:
        Nx2 float array of contour points
    """
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    
    # Clean small holes / noise with adaptive kernel size based on image size
    h, w = mask_u8.shape
    kernel_size = max(3, min(7, min(h, w) // 100))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Close to fill small gaps, then open to remove small noise
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Optional: slight erosion followed by dilation to smooth jagged edges
    if smooth and kernel_size >= 3:
        smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_u8 = cv2.erode(mask_u8, smooth_kernel, iterations=1)
        mask_u8 = cv2.dilate(mask_u8, smooth_kernel, iterations=1)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("no contour found — try a clearer subject or --method rembg")
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 50:
        raise ValueError("contour too small — subject may not have been segmented")
    pts = contour.reshape(-1, 2).astype(float)
    return pts


def _mask_from_alpha(img_bgra: np.ndarray) -> np.ndarray | None:
    if img_bgra.ndim == 3 and img_bgra.shape[2] == 4:
        return img_bgra[:, :, 3]
    return None


def _mask_otsu(bgr: np.ndarray, invert: bool | None = None) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Prefer the minority / non-border class as the subject when possible
    if invert is None:
        h, w = mask.shape
        border = np.concatenate(
            [mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]]
        )
        border_white = np.mean(border > 127) > 0.5
        # If border is mostly white, subject is likely dark → invert
        invert = border_white
    if invert:
        mask = cv2.bitwise_not(mask)
    return mask


def _mask_grabcut(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    # Assume subject is roughly in the central 80% of the frame
    margin_x, margin_y = int(0.1 * w), int(0.1 * h)
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(
        np.uint8
    )


def _mask_canny(bgr: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """Use Canny edge detection - captures actual line art, not filled regions."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, low, high)
    # Dilate to connect nearby edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    return edges


def _skeletonize_opencv(binary: np.ndarray) -> np.ndarray:
    """
    Zhang-Suen thinning using OpenCV morphological operations.
    Produces 1-pixel wide skeleton of the input binary image.
    """
    skeleton = np.zeros_like(binary)
    img = binary.copy()
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()
        
        if cv2.countNonZero(img) == 0:
            break
    
    return skeleton


def _extract_skeleton(bgr: np.ndarray, invert: bool = True) -> np.ndarray:
    """
    Extract the skeleton (thin line) of line art using morphological thinning.
    This captures ALL lines in a drawing, not just the outer boundary.
    
    For drawings with dark lines on light background, invert=True (default).
    For drawings with light lines on dark background, invert=False.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    # Adaptive thresholding works better for varied lighting
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
    )
    
    if not invert:
        binary = 255 - binary
    
    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Thin to skeleton (1-pixel wide lines) using OpenCV
    skeleton = _skeletonize_opencv(binary)
    
    return skeleton


def _skeleton_to_points(skeleton: np.ndarray) -> np.ndarray:
    """
    Extract all skeleton pixel coordinates.
    Returns Nx2 array of (x, y) coordinates.
    """
    ys, xs = np.where(skeleton > 0)
    return np.column_stack([xs, ys]).astype(float)


def _build_pixel_graph(skeleton: np.ndarray) -> dict:
    """
    Build an adjacency graph from skeleton pixels.
    Each pixel is connected to its 8-connected neighbors.
    """
    h, w = skeleton.shape
    ys, xs = np.where(skeleton > 0)
    
    # Map (x,y) -> index
    coord_to_idx = {}
    for i, (x, y) in enumerate(zip(xs, ys)):
        coord_to_idx[(x, y)] = i
    
    # Build adjacency list
    adj = defaultdict(list)
    for x, y in zip(xs, ys):
        idx = coord_to_idx[(x, y)]
        # Check 8-connected neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if (nx, ny) in coord_to_idx:
                    adj[idx].append(coord_to_idx[(nx, ny)])
    
    return adj, np.column_stack([xs, ys]).astype(float)


def _trace_skeleton_dfs(adj: dict, points: np.ndarray, start: int) -> np.ndarray:
    """
    Trace skeleton pixels in local stroke order.

    A depth-first traversal is a poor fit for drawing: when it reaches a
    junction, the DFS stack jumps to another branch and the resulting
    sequence contains long diagonal segments. Follow the most forward
    unvisited neighbor instead. When a stroke ends, jump to the nearest
    remaining pixel (the unavoidable connector between separate strokes).
    """
    n = len(points)
    if n == 0:
        return points
    visited = np.zeros(n, dtype=bool)
    path: list[int] = []
    tree = KDTree(points)
    current = int(start)
    previous: int | None = None

    while len(path) < n:
        if visited[current]:
            # Start the next disconnected/untraced branch at its nearest pixel.
            unvisited = np.flatnonzero(~visited)
            if not len(unvisited):
                break
            _, candidates = tree.query(points[current], k=min(64, n))
            candidates = np.atleast_1d(candidates)
            next_start = next((int(i) for i in candidates if not visited[int(i)]), None)
            current = next_start if next_start is not None else int(unvisited[0])
            previous = None

        visited[current] = True
        path.append(current)
        candidates = [i for i in adj.get(current, ()) if not visited[i]]

        if candidates and previous is not None:
            incoming = points[current] - points[previous]
            incoming_norm = np.linalg.norm(incoming)
            if incoming_norm > 0:
                # Prefer continuing straight through junctions; a tiny
                # distance term makes the choice stable for diagonal pixels.
                def continuation_score(i: int) -> float:
                    outgoing = points[i] - points[current]
                    norm = np.linalg.norm(outgoing)
                    alignment = float(np.dot(incoming, outgoing) / (incoming_norm * norm))
                    return alignment - 0.01 * norm

                nxt = max(candidates, key=continuation_score)
            else:
                nxt = min(candidates, key=lambda i: np.linalg.norm(points[i] - points[current]))
        elif candidates:
            nxt = min(candidates, key=lambda i: np.linalg.norm(points[i] - points[current]))
        else:
            # Force the outer loop to select the nearest unvisited pixel.
            previous = None
            current = next((int(i) for i in tree.query(points[current], k=min(64, n))[1]
                            if not visited[int(i)]), current)
            continue

        previous, current = current, int(nxt)

    return points[np.asarray(path, dtype=int)]


def _order_points_tsp(points: np.ndarray, start_idx: int = 0) -> np.ndarray:
    """
    Order points using nearest-neighbor heuristic (greedy TSP).
    This creates a continuous path through all points.
    
    Much faster than exact TSP for large point sets.
    """
    if len(points) < 3:
        return points
    
    n = len(points)
    visited = np.zeros(n, dtype=bool)
    order = [start_idx]
    visited[start_idx] = True
    
    # Build KD-tree for efficient nearest neighbor lookup
    tree = KDTree(points)
    
    current = start_idx
    for _ in range(n - 1):
        # Find nearest unvisited point
        # Query more candidates to find unvisited ones faster
        dists, indices = tree.query(points[current], k=min(50, n))
        
        found = False
        for idx in indices:
            if not visited[idx]:
                visited[idx] = True
                order.append(idx)
                current = idx
                found = True
                break
        
        if not found:
            # Fallback: find any unvisited point
            for i in range(n):
                if not visited[i]:
                    visited[i] = True
                    order.append(i)
                    current = i
                    break
    
    return points[order]


def _order_by_skeleton_trace(skeleton: np.ndarray) -> np.ndarray:
    """
    Order skeleton points by tracing along connected edges.
    This follows the natural structure of line art.
    """
    adj, points = _build_pixel_graph(skeleton)
    
    if len(points) < 3:
        return points
    
    # Find a good starting point (endpoint with degree 1, or corner)
    degrees = {i: len(adj.get(i, [])) for i in range(len(points))}
    endpoints = [i for i, d in degrees.items() if d == 1]
    
    if endpoints:
        # Start from the topmost endpoint
        start = min(endpoints, key=lambda i: (points[i][1], points[i][0]))
    else:
        # Start from top-left corner of bounding box
        min_y = points[:, 1].min()
        top_row = np.where(points[:, 1] < min_y + 5)[0]
        start = int(top_row[np.argmin(points[top_row, 0])]) if len(top_row) > 0 else 0
    
    return _trace_skeleton_dfs(adj, points, start)


def _subsample_points(points: np.ndarray, max_points: int = 10000) -> np.ndarray:
    """
    Subsample points evenly if there are too many.
    Keeps the first and last points.
    """
    n = len(points)
    if n <= max_points:
        return points
    
    # Keep every nth point
    step = n // max_points
    indices = np.arange(0, n, step)
    
    return points[indices]


def _smooth_path(points: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Apply moving average smoothing to the path.
    Reduces jaggedness from pixel-level ordering.
    """
    if len(points) < window:
        return points
    
    # Pad for circular smoothing
    padded = np.vstack([points[-window//2:], points, points[:window//2]])
    smoothed = np.zeros_like(points)
    
    for i in range(len(points)):
        smoothed[i] = np.mean(padded[i:i+window], axis=0)
    
    return smoothed


def extract_line_art_path(
    image_path: str | Path,
    n_points: int = 4096,
    invert: bool = True,
    smooth: int = 5,
    use_skeleton_trace: bool = True,
    debug_path: str | Path | None = None,
) -> np.ndarray:
    """
    Extract ALL lines from a drawing as a single continuous path.
    
    This is the key function for complex line-art drawings!
    
    Unlike extract_path_from_image which only gets the outer contour,
    this function:
    1. Extracts the skeleton of ALL lines in the image
    2. Orders them by tracing along connected edges (reduces jumps)
    3. Returns a single continuous path that traces all details
    
    Args:
        image_path: Path to input image (line art drawing)
        n_points: Number of output points (more = higher detail)
        invert: True for dark-on-light drawings, False for light-on-dark
        smooth: Smoothing window size (0 to disable)
        use_skeleton_trace: If True, trace along skeleton edges (fewer jumps)
        debug_path: Optional path to save debug visualization
    
    Returns:
        Nx2 float array of path points (centered, y-up math coords)
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not read image: {path}")
    
    # Extract skeleton of all lines
    skeleton = _extract_skeleton(img, invert=invert)
    
    ys, xs = np.where(skeleton > 0)
    n_pixels = len(xs)
    
    if n_pixels < 10:
        raise ValueError("No line art detected - try inverting or use a clearer drawing")
    
    print(f"  Skeleton: {n_pixels} pixels")
    
    # Order points by tracing along skeleton edges
    if use_skeleton_trace:
        print(f"  Tracing skeleton edges...")
        ordered = _order_by_skeleton_trace(skeleton)
    else:
        # Fall back to TSP
        raw_points = np.column_stack([xs, ys]).astype(float)
        min_x, min_y = raw_points.min(axis=0)
        dists_to_corner = (raw_points[:, 0] - min_x)**2 + (raw_points[:, 1] - min_y)**2
        start_idx = int(np.argmin(dists_to_corner))
        print(f"  Ordering points (TSP)...")
        ordered = _order_points_tsp(raw_points, start_idx)
    
    # Subsample to target resolution
    if len(ordered) > n_points * 2:
        ordered = _subsample_points(ordered, n_points * 2)
    
    # Smooth the path
    if smooth > 0:
        ordered = _smooth_path(ordered, window=smooth)
    
    # Resample to exact n_points
    pts = _resample_closed(ordered, n_points)
    
    # Convert to math coords (y up, centered)
    pts[:, 1] = -pts[:, 1]
    pts -= pts.mean(axis=0)
    
    # Debug output
    if debug_path is not None:
        dbg = img.copy()
        skeleton_color = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
        skeleton_color[skeleton > 0] = [0, 255, 0]
        overlay = cv2.addWeighted(dbg, 0.5, skeleton_color, 0.5, 0)
        cv2.imwrite(str(debug_path), overlay)
        print(f"  Debug saved → {debug_path}")
    
    return pts


def extract_all_contours_path(
    image_path: str | Path,
    n_points: int = 4096,
    method: str = "canny",
    debug_path: str | Path | None = None,
) -> np.ndarray:
    """
    Extract ALL contours (not just external) and connect them into one path.
    
    This handles images with separate shapes/components by connecting
    all contours into a single traversable path.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"could not read image: {path}")
    
    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        bgr = img
    
    # Get edges
    if method == "canny":
        mask = _mask_canny(bgr)
    else:
        # Try alpha channel first, fall back to Otsu
        alpha = _mask_from_alpha(img) if img.ndim == 3 and img.shape[2] == 4 else None
        mask = alpha if alpha is not None else _mask_otsu(bgr)
    
    # Find ALL contours including internal ones
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        raise ValueError("No contours found")
    
    # Collect all contour points
    all_points = []
    for contour in contours:
        if cv2.contourArea(contour) > 20:  # Skip tiny noise
            pts = contour.reshape(-1, 2).astype(float)
            all_points.append(pts)
    
    if not all_points:
        raise ValueError("No significant contours found")
    
    print(f"  Found {len(all_points)} contours")
    
    # Connect contours: end of one → start of nearest next
    connected = []
    remaining = list(range(len(all_points)))
    current_idx = 0  # Start with first contour
    
    while remaining:
        remaining.remove(current_idx)
        connected.append(all_points[current_idx])
        
        if not remaining:
            break
        
        # Find nearest contour to continue
        end_point = all_points[current_idx][-1]
        min_dist = float('inf')
        next_idx = remaining[0]
        
        for idx in remaining:
            # Check distance to both start and end of candidate contour
            start = all_points[idx][0]
            dist = np.hypot(end_point[0] - start[0], end_point[1] - start[1])
            if dist < min_dist:
                min_dist = dist
                next_idx = idx
        
        current_idx = next_idx
    
    # Concatenate all connected contours
    combined = np.vstack(connected)
    print(f"  Combined: {len(combined)} points")
    
    # Resample
    pts = _resample_closed(combined, n_points)
    
    # Convert to math coords
    pts[:, 1] = -pts[:, 1]
    pts -= pts.mean(axis=0)
    
    if debug_path is not None:
        dbg = bgr.copy()
        cv2.drawContours(dbg, contours, -1, (0, 255, 0), 1)
        cv2.imwrite(str(debug_path), dbg)
        print(f"  Debug saved → {debug_path}")
    
    return pts


def _mask_rembg(bgr: np.ndarray) -> np.ndarray:
    try:
        from rembg import remove
    except ImportError as exc:
        raise ImportError(
            "rembg is not installed. Run: pip install rembg\n"
            "Or use --method otsu / --method grabcut"
        ) from exc
    rgba = remove(bgr)
    if isinstance(rgba, bytes):
        arr = np.frombuffer(rgba, dtype=np.uint8)
        rgba = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if rgba is None:
        raise ValueError("rembg failed to segment the image")
    if rgba.ndim == 3 and rgba.shape[2] >= 4:
        return rgba[:, :, 3]
    gray = cv2.cvtColor(rgba, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    return mask


def extract_path_from_image(
    image_path: str | Path,
    n_points: int = 512,
    method: str = "auto",
    invert: bool | None = None,
    simplify: float = 0.0,  # Default to NO simplification for maximum detail
    smooth_contour: bool = False,  # Disable smoothing for complex shapes
    debug_path: str | Path | None = None,
) -> np.ndarray:
    """
    Load an image and return a closed outline as (n_points, 2) float array.

    Coordinates are in a math-friendly frame: x right, y up, origin at center
    of the contour bounding box (not yet globally normalized).
    
    Methods:
        auto:     Auto-detect best method based on image properties
        alpha:    Use alpha channel as mask
        otsu:     Otsu thresholding (good for silhouettes)
        grabcut:  GrabCut segmentation (good for photos)
        rembg:    AI background removal (requires rembg package)
        canny:    Canny edge detection
        lineart:  Classical CV skeleton + TSP (all ink strokes)
        allcontours: Extract and connect ALL contours (internal + external)
        ai:       Neural lineart detection + closed tour
        ai_anime: Neural character lineart detection
    """
    method = method.lower()
    
    # Special handling for line art extraction - completely different approach
    if method == "lineart":
        return extract_line_art_path(
            image_path, 
            n_points=n_points, 
            invert=invert if invert is not None else True,
            debug_path=debug_path
        )
    
    if method == "allcontours":
        return extract_all_contours_path(
            image_path,
            n_points=n_points,
            debug_path=debug_path
        )

    if method in ("ai", "ai_anime", "controlnet"):
        from .ai_path import extract_ai_path
        style = "anime" if method == "ai_anime" else "realistic"
        return extract_ai_path(
            image_path,
            n_points=n_points,
            style=style,
            coarse=False,
            use_rembg=False,
            debug_path=debug_path,
        )
    
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"could not read image: {path}")

    alpha = _mask_from_alpha(img)
    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        bgr = img

    if method == "boundary":
        pts, _, _, _, _ = extract_boundary_contour_path(
            bgr, n_points=n_points, invert=bool(invert)
        )
        pts = pts.copy()
        pts[:, 1] = -pts[:, 1]
        pts -= pts.mean(axis=0)
        return pts

    if method == "auto":
        if alpha is not None and np.any(alpha > 0):
            mask = alpha
            method_used = "alpha"
        else:
            # Photos: try grabcut; flat silhouettes often work with otsu
            try:
                mask = _mask_grabcut(bgr)
                method_used = "grabcut"
            except Exception:
                mask = _mask_otsu(bgr, invert=invert)
                method_used = "otsu"
    elif method == "alpha":
        if alpha is None:
            raise ValueError("image has no alpha channel")
        mask = alpha
        method_used = "alpha"
    elif method == "otsu":
        mask = _mask_otsu(bgr, invert=invert)
        method_used = "otsu"
    elif method == "grabcut":
        mask = _mask_grabcut(bgr)
        method_used = "grabcut"
    elif method == "rembg":
        mask = _mask_rembg(bgr)
        method_used = "rembg"
    elif method == "canny":
        mask = _mask_canny(bgr)
        method_used = "canny"
    elif method == "adaptive":
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV if not invert else cv2.THRESH_BINARY,
            21, 10
        )
        method_used = "adaptive"
    else:
        raise ValueError(f"unknown method: {method}")

    pts = _largest_external_contour(mask, smooth=smooth_contour)

    # Optional polyline simplification (keeps shape, fewer jagged pixels)
    if simplify and simplify > 0:
        peri = cv2.arcLength(pts.astype(np.float32).reshape(-1, 1, 2), True)
        approx = cv2.approxPolyDP(
            pts.astype(np.float32).reshape(-1, 1, 2), simplify * peri, True
        )
        if len(approx) >= 8:
            pts = approx.reshape(-1, 2).astype(float)

    pts = _resample_closed(pts, n_points)

    # Image coords → math coords (y up), centered on contour
    pts[:, 1] = -pts[:, 1]
    pts -= pts.mean(axis=0)

    if debug_path is not None:
        dbg = bgr.copy()
        # Re-project for overlay (undo y flip + centering approx via bbox)
        # Draw mask outline on original for user feedback
        mask_u8 = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(dbg, contours, -1, (0, 165, 255), 2)
        cv2.imwrite(str(debug_path), dbg)
        print(f"Contour debug ({method_used}) saved → {debug_path}")

    return pts


def save_silhouette_preview(
    points: np.ndarray,
    out_path: str | Path,
    size: int = 512,
    fill: bool = True,
) -> None:
    """Save a simple filled silhouette PNG of the extracted path (for left panel)."""
    pts = np.asarray(points, dtype=float).copy()
    pts -= pts.mean(axis=0)
    span = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 1e-9)
    pts *= (size * 0.8) / span
    pts[:, 0] += size / 2
    pts[:, 1] = size / 2 - pts[:, 1]  # y up → image y down

    canvas = np.zeros((size, size, 4), dtype=np.uint8)
    poly = pts.astype(np.int32).reshape(-1, 1, 2)
    if fill:
        cv2.fillPoly(canvas, [poly], (40, 30, 20, 255))
    cv2.polylines(canvas, [poly], True, (80, 90, 230, 255), 3, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)
