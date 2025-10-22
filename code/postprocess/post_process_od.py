"""
frame_embeddings.py

Builds frame-level embeddings from object-detection CSV output.

Expected CSV format (header):
frame,class,confidence,xmin,ymin,xmax,ymax

Defaults:
 - grid: 10x10
 - classes: ['adult','chick','fish']
 - fps: 30
 - conf_thresh: 0.25
 - smoothing window: 30 seconds (default)
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.decomposition import PCA
from scipy.linalg import pinv
import matplotlib.pyplot as plt
from datetime import datetime
import os
import re

# Optional: tqdm for progress (pip install tqdm)
try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x: x

##############################
# Utility / IO
##############################

def parse_start_time_from_filename(csv_path, fps=25):
    """
    Parse start time from filename like 'Auklab1_FAR3_2024-06-17_05.00.00_raw.csv'
    Returns datetime object or None if parsing fails
    """
    filename = os.path.basename(csv_path)
    
    # Pattern to match: YYYY-MM-DD_HH.MM.SS
    pattern = r'(\d{4}-\d{2}-\d{2})_(\d{2})\.(\d{2})\.(\d{2})'
    match = re.search(pattern, filename)
    
    if match:
        date_str = match.group(1)  # YYYY-MM-DD
        hour = match.group(2)
        minute = match.group(3) 
        second = match.group(4)
        
        # Create datetime string
        datetime_str = f"{date_str} {hour}:{minute}:{second}"
        try:
            start_time = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
            return start_time
        except ValueError:
            print(f"Warning: Could not parse datetime from {datetime_str}")
            return None
    else:
        print(f"Warning: Could not extract timestamp from filename {filename}")
        return None

def add_time_column(per_frame_summary, start_time=None, fps=25):
    """
    Add time column to per_frame_summary DataFrame.
    If start_time is provided, creates absolute timestamps.
    Otherwise creates relative time in seconds from frame 0.
    """
    df = per_frame_summary.copy()
    
    # Calculate relative time in seconds
    df['time_seconds'] = df['frame'] / fps
    
    if start_time is not None:
        # Add absolute timestamps
        df['timestamp'] = pd.to_datetime(start_time) + pd.to_timedelta(df['time_seconds'], unit='s')
        # Use timestamp as the main time column for plotting
        df['time'] = df['timestamp']
    else:
        # Use relative time in seconds
        df['time'] = df['time_seconds']
    
    return df

def load_detections(csv_path, conf_thresh=0.25):
    """
    Load CSV and filter by confidence.
    Returns pandas DataFrame with columns:
    ['frame','class','confidence','xmin','ymin','xmax','ymax']
    """
    df = pd.read_csv(csv_path)
    # Ensure numeric types
    for c in ['frame','confidence','xmin','ymin','xmax','ymax']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['frame','xmin','ymin','xmax','ymax','confidence'])
    df = df[df['confidence'] >= conf_thresh].copy()
    df['frame'] = df['frame'].astype(int)
    return df

def infer_frame_size(df, margin=1.05):
    """
    Infer frame width and height from max box coordinates (if user doesn't supply).
    margin multiplies inferred size slightly to avoid edge clamping.
    """
    max_x = df['xmax'].max()
    max_y = df['ymax'].max()
    if np.isnan(max_x) or np.isnan(max_y):
        raise ValueError("Can't infer frame size from empty dataframe.")
    width = int(np.ceil(max_x * margin))
    height = int(np.ceil(max_y * margin))
    return width, height

##############################
# Adaptive Grid Generation
##############################

def sample_object_positions(df, n_samples=1000, random_seed=42):
    """
    Sample object center positions from random frames to create an adaptive grid.
    Returns array of (x, y) coordinates.
    """
    np.random.seed(random_seed)
    
    if len(df) == 0:
        return np.array([]).reshape(0, 2)
    
    # Sample up to n_samples detections randomly
    sample_size = min(n_samples, len(df))
    sampled_df = df.sample(n=sample_size, random_state=random_seed)
    
    # Calculate center positions
    centers = []
    for _, row in sampled_df.iterrows():
        cx = (row['xmin'] + row['xmax']) / 2.0
        cy = (row['ymin'] + row['ymax']) / 2.0
        centers.append([cx, cy])
    
    return np.array(centers)

def create_adaptive_grid(positions, frame_width, frame_height, target_cells=64, min_points_per_cell=5,
                        min_cell_size=None, max_aspect_ratio=None):
    """
    Create an adaptive grid based on object positions using recursive binary splits.
    
    Args:
        positions: Nx2 array of (x, y) coordinates
        frame_width, frame_height: Frame dimensions
        target_cells: Approximate number of cells desired
        min_points_per_cell: Minimum points per cell to allow further splitting
        min_cell_size: Minimum cell width/height (fraction of frame if ≤1, pixels if >1)
        max_aspect_ratio: Maximum length/width ratio (e.g., 4.0 = 4:1 max)
    
    Returns:
        list of dicts with grid cell boundaries: [{'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2, 'points': N}, ...]
    """
    if len(positions) == 0:
        # Return single cell covering entire frame
        return [{'xmin': 0, 'ymin': 0, 'xmax': frame_width, 'ymax': frame_height, 'points': 0}]
    
    # Convert min_cell_size to absolute pixels if specified as fraction
    if min_cell_size is not None:
        if min_cell_size <= 1.0:  # Assume fraction of frame
            min_width = min_cell_size * frame_width
            min_height = min_cell_size * frame_height
        else:  # Assume absolute pixels
            min_width = min_height = min_cell_size
    else:
        min_width = min_height = 0
    
    def can_split_cell(cell, split_direction):
        """Check if a cell can be split without violating size/aspect ratio constraints."""
        width = cell['xmax'] - cell['xmin']
        height = cell['ymax'] - cell['ymin']
        
        if split_direction == 'vertical':
            # Splitting vertically creates two cells with width/2
            new_width = width / 2
            new_height = height
        else:  # horizontal
            # Splitting horizontally creates two cells with height/2
            new_width = width
            new_height = height / 2
        
        # Check minimum size constraint
        if new_width < min_width or new_height < min_height:
            return False
        
        # Check aspect ratio constraint
        if max_aspect_ratio is not None:
            aspect_ratio = max(new_width / new_height, new_height / new_width)
            if aspect_ratio > max_aspect_ratio:
                return False
        
        return True
    
    # Start with the entire frame as one cell
    cells = [{'xmin': 0, 'ymin': 0, 'xmax': frame_width, 'ymax': frame_height, 
              'positions': positions}]
    
    # Recursively split cells until we reach target number or can't split further
    while len(cells) < target_cells:
        # Find the cell with the most points that can be split
        best_cell_idx = -1
        max_points = 0
        
        for i, cell in enumerate(cells):
            if len(cell['positions']) > max_points and len(cell['positions']) >= min_points_per_cell * 2:
                max_points = len(cell['positions'])
                best_cell_idx = i
        
        if best_cell_idx == -1:
            # No more cells can be split based on point requirements
            break
        
        # Check if the best cell can be split given our constraints
        cell = cells[best_cell_idx]
        pos = cell['positions']
        
        # Decide whether to split horizontally or vertically
        width = cell['xmax'] - cell['xmin']
        height = cell['ymax'] - cell['ymin']
        
        # Try preferred direction first (split along longer dimension)
        split_vertically = width > height
        split_direction = 'vertical' if split_vertically else 'horizontal'
        
        # Check if we can split in the preferred direction
        if not can_split_cell(cell, split_direction):
            # Try the other direction
            split_direction = 'horizontal' if split_vertically else 'vertical'
            split_vertically = not split_vertically
            
            if not can_split_cell(cell, split_direction):
                # Can't split in either direction due to constraints
                break
        
        if split_vertically:
            # Split vertically (along x-axis)
            # Find median x-coordinate
            x_coords = pos[:, 0]
            split_x = np.median(x_coords)
            
            left_mask = pos[:, 0] <= split_x
            right_mask = pos[:, 0] > split_x
            
            left_cell = {'xmin': cell['xmin'], 'ymin': cell['ymin'], 
                        'xmax': split_x, 'ymax': cell['ymax'],
                        'positions': pos[left_mask]}
            right_cell = {'xmin': split_x, 'ymin': cell['ymin'],
                         'xmax': cell['xmax'], 'ymax': cell['ymax'],
                         'positions': pos[right_mask]}
        else:
            # Split horizontally (along y-axis)
            # Find median y-coordinate
            y_coords = pos[:, 1]
            split_y = np.median(y_coords)
            
            top_mask = pos[:, 1] <= split_y
            bottom_mask = pos[:, 1] > split_y
            
            top_cell = {'xmin': cell['xmin'], 'ymin': cell['ymin'],
                       'xmax': cell['xmax'], 'ymax': split_y,
                       'positions': pos[top_mask]}
            bottom_cell = {'xmin': cell['xmin'], 'ymin': split_y,
                          'xmax': cell['xmax'], 'ymax': cell['ymax'],
                          'positions': pos[bottom_mask]}
            
            left_cell, right_cell = top_cell, bottom_cell
        
        # Verify both new cells meet minimum point requirements AND constraints
        if (len(left_cell['positions']) < min_points_per_cell or 
            len(right_cell['positions']) < min_points_per_cell):
            # Can't split due to insufficient points in resulting cells
            break
        
        # Verify the actual resulting cells meet our size and aspect ratio constraints
        left_width = left_cell['xmax'] - left_cell['xmin']
        left_height = left_cell['ymax'] - left_cell['ymin']
        right_width = right_cell['xmax'] - right_cell['xmin']
        right_height = right_cell['ymax'] - right_cell['ymin']
        
        # Check size constraints
        if (left_width < min_width or left_height < min_height or
            right_width < min_width or right_height < min_height):
            break
        
        # Check aspect ratio constraints
        if max_aspect_ratio is not None:
            left_aspect = max(left_width / left_height, left_height / left_width)
            right_aspect = max(right_width / right_height, right_height / right_width)
            if left_aspect > max_aspect_ratio or right_aspect > max_aspect_ratio:
                break
        
        # Replace the original cell with the two new cells
        cells = cells[:best_cell_idx] + [left_cell, right_cell] + cells[best_cell_idx+1:]
    
    # Clean up - remove positions from final cells and add point count
    final_cells = []
    for cell in cells:
        final_cell = {
            'xmin': cell['xmin'], 'ymin': cell['ymin'],
            'xmax': cell['xmax'], 'ymax': cell['ymax'],
            'points': len(cell['positions'])
        }
        final_cells.append(final_cell)
    
    return final_cells

def save_adaptive_grid(grid_cells, filepath):
    """Save adaptive grid to JSON file."""
    import json
    with open(filepath, 'w') as f:
        json.dump(grid_cells, f, indent=2)

def load_adaptive_grid(filepath):
    """Load adaptive grid from JSON file."""
    import json
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_adaptive_grid(grid_cells, frame_width, frame_height, positions=None, title="Adaptive Grid"):
    """
    Plot the adaptive grid boundaries, optionally with object positions.
    """
    plt.figure(figsize=(12, 8))
    
    # Plot grid boundaries
    for cell in grid_cells:
        rect = plt.Rectangle((cell['xmin'], cell['ymin']), 
                           cell['xmax'] - cell['xmin'], 
                           cell['ymax'] - cell['ymin'],
                           fill=False, edgecolor='red', linewidth=1)
        plt.gca().add_patch(rect)
        
        # Add point count in center of cell
        cx = (cell['xmin'] + cell['xmax']) / 2
        cy = (cell['ymin'] + cell['ymax']) / 2
        plt.text(cx, cy, str(cell['points']), 
                ha='center', va='center', fontsize=8, 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    # Plot object positions if provided
    if positions is not None and len(positions) > 0:
        plt.scatter(positions[:, 0], positions[:, 1], 
                   c='blue', s=10, alpha=0.6, label='Object centers')
        plt.legend()
    
    plt.xlim(0, frame_width)
    plt.ylim(0, frame_height)
    plt.gca().invert_yaxis()  # Image coordinates (y increases downward)
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.title(f'{title} ({len(grid_cells)} cells)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

##############################
# NMS (per-class, per-frame)
##############################

def _iou(box, boxes):
    """
    box: [xmin,ymin,xmax,ymax]
    boxes: Nx4 array
    returns IoU vector
    """
    xa = np.maximum(box[0], boxes[:,0])
    ya = np.maximum(box[1], boxes[:,1])
    xb = np.minimum(box[2], boxes[:,2])
    yb = np.minimum(box[3], boxes[:,3])
    inter_w = np.maximum(0.0, xb - xa)
    inter_h = np.maximum(0.0, yb - ya)
    inter = inter_w * inter_h
    area1 = (box[2]-box[0])*(box[3]-box[1])
    area2 = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    union = area1 + area2 - inter
    return inter / (union + 1e-8)

def nms_boxes(boxes, scores, iou_thresh=0.5):
    """
    Simple NMS on boxes for single frame + class.
    boxes: Nx4 numpy
    scores: N
    returns indices kept
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)
    idxs = scores.argsort()[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        ious = _iou(boxes[i], boxes[rest])
        idxs = rest[ious <= iou_thresh]
    return np.array(keep, dtype=int)

def apply_nms(df, iou_thresh=0.5):
    """
    Apply per-class, per-frame NMS. Returns filtered DataFrame.
    """
    kept_rows = []
    for (frame, cls), group in tqdm(df.groupby(['frame','class'])):
        boxes = group[['xmin','ymin','xmax','ymax']].to_numpy()
        scores = group['confidence'].to_numpy()
        keep_idx = nms_boxes(boxes, scores, iou_thresh=iou_thresh)
        kept = group.iloc[keep_idx]
        kept_rows.append(kept)
    if len(kept_rows) == 0:
        return df.iloc[0:0].copy()
    return pd.concat(kept_rows, ignore_index=True)

##############################
# Embedding construction
##############################

def compute_frame_embeddings(df,
                             grid_size=(10,10),
                             classes=('adult','chick','fish'),
                             frame_width=None,
                             frame_height=None,
                             apply_nms_flag=False,
                             iou_thresh=0.5,
                             normalize_by_frame_area=False,
                             adaptive_grid=None,
                             grid_cells=None):
    """
    Build frame embeddings using either uniform grid or adaptive grid:
      per-cell vector: [count_adult, count_chick, count_fish, total_bbox_area, mean_confidence]
    
    Args:
        df: DataFrame with detections
        grid_size: (w, h) for uniform grid (ignored if using adaptive)
        adaptive_grid: dict with adaptive grid parameters or None for uniform grid
        grid_cells: pre-computed grid cells or None to compute new
        
    Returns:
      embeddings: np.array shape (n_frames, n_cells*5)
      frames_index: array/list of frame numbers (0..max_frame)
      per_frame_summary_df: DataFrame with total counts and sums for quick plotting
      grid_cells: the grid cells used (for saving/reuse)
      raw_cells: raw cell data for visualization
    """
    if apply_nms_flag:
        df = apply_nms(df, iou_thresh=iou_thresh)

    if frame_width is None or frame_height is None:
        frame_width, frame_height = infer_frame_size(df)

    # Determine if using adaptive grid
    use_adaptive = adaptive_grid is not None or grid_cells is not None
    
    if use_adaptive:
        if grid_cells is None:
            # Generate new adaptive grid
            print("Generating adaptive grid...")
            positions = sample_object_positions(df, 
                                              n_samples=adaptive_grid.get('n_samples', 1000),
                                              random_seed=adaptive_grid.get('random_seed', 42))
            grid_cells = create_adaptive_grid(positions, frame_width, frame_height,
                                            target_cells=adaptive_grid.get('target_cells', 64),
                                            min_points_per_cell=adaptive_grid.get('min_points_per_cell', 5),
                                            min_cell_size=adaptive_grid.get('min_cell_size', None),
                                            max_aspect_ratio=adaptive_grid.get('max_aspect_ratio', None))
            print(f"Generated adaptive grid with {len(grid_cells)} cells")
        else:
            print(f"Using provided adaptive grid with {len(grid_cells)} cells")
        
        return _compute_adaptive_embeddings(df, grid_cells, classes, frame_width, frame_height, normalize_by_frame_area)
    else:
        # Use original uniform grid method
        return _compute_uniform_embeddings(df, grid_size, classes, frame_width, frame_height, normalize_by_frame_area)

def _compute_uniform_embeddings(df, grid_size, classes, frame_width, frame_height, normalize_by_frame_area):
    """Original uniform grid embedding computation."""
    grid_w, grid_h = grid_size
    cell_w = frame_width / grid_w
    cell_h = frame_height / grid_h

    # Get unique frame numbers that actually exist in the data
    if len(df) > 0:
        unique_frames = sorted(df['frame'].unique())
        frame_to_idx = {frame: i for i, frame in enumerate(unique_frames)}
        n_frames = len(unique_frames)
    else:
        unique_frames = []
        frame_to_idx = {}
        n_frames = 0

    # raw cells: frames x rows(y) x cols(x) x features(5)
    raw = np.zeros((n_frames, grid_h, grid_w, 5), dtype=float)
    conf_counts = np.zeros((n_frames, grid_h, grid_w), dtype=int)
    conf_sums = np.zeros((n_frames, grid_h, grid_w), dtype=float)

    class_to_idx = {c: i for i, c in enumerate(classes)}

    for r in df.itertuples(index=False):
        fr = int(r.frame)
        fr_idx = frame_to_idx[fr]  # Map frame number to array index
        cx = (r.xmin + r.xmax) / 2.0
        cy = (r.ymin + r.ymax) / 2.0
        col = int(np.clip(cx // cell_w, 0, grid_w - 1))
        row = int(np.clip(cy // cell_h, 0, grid_h - 1))
        
        cls = r._1  # pandas renames 'class' column to '_1' in itertuples
        if cls in class_to_idx:
            raw[fr_idx, row, col, class_to_idx[cls]] += 1.0
        
        area = max(0.0, (r.xmax - r.xmin) * (r.ymax - r.ymin))
        raw[fr_idx, row, col, 3] += area
        conf_sums[fr_idx, row, col] += r.confidence
        conf_counts[fr_idx, row, col] += 1

    # Compute mean confidence
    nonzero = conf_counts > 0
    raw[nonzero, 4] = conf_sums[nonzero] / conf_counts[nonzero]

    if normalize_by_frame_area:
        frame_area = frame_width * frame_height
        raw[..., 3] = raw[..., 3] / (frame_area + 1e-12)

    # Flatten per-frame
    embeddings = raw.reshape(n_frames, grid_h * grid_w * 5)

    # Create per-frame summary
    total_counts = raw[..., 0:3].sum(axis=(1,2))
    total_bbox_area = raw[..., 3].sum(axis=(1,2))
    mean_conf = np.zeros(n_frames)
    total_conf_count = conf_counts.sum(axis=(1,2))
    total_conf_sum = conf_sums.sum(axis=(1,2))
    mean_conf[total_conf_count > 0] = total_conf_sum[total_conf_count > 0] / total_conf_count[total_conf_count > 0]

    per_frame_summary = pd.DataFrame({
        'frame': unique_frames,
        'count_adult': total_counts[:,0],
        'count_chick': total_counts[:,1],
        'count_fish' : total_counts[:,2],
        'total_bbox_area': total_bbox_area,
        'mean_confidence': mean_conf
    })

    return embeddings, np.array(unique_frames), per_frame_summary, None, raw

def _compute_adaptive_embeddings(df, grid_cells, classes, frame_width, frame_height, normalize_by_frame_area):
    """Adaptive grid embedding computation."""
    # Get unique frame numbers that actually exist in the data
    if len(df) > 0:
        unique_frames = sorted(df['frame'].unique())
        frame_to_idx = {frame: i for i, frame in enumerate(unique_frames)}
        n_frames = len(unique_frames)
    else:
        unique_frames = []
        frame_to_idx = {}
        n_frames = 0
    
    n_cells = len(grid_cells)

    # embeddings: frames x cells x features(5)
    raw = np.zeros((n_frames, n_cells, 5), dtype=float)
    conf_counts = np.zeros((n_frames, n_cells), dtype=int)
    conf_sums = np.zeros((n_frames, n_cells), dtype=float)

    class_to_idx = {c: i for i, c in enumerate(classes)}

    for r in df.itertuples(index=False):
        fr = int(r.frame)
        fr_idx = frame_to_idx[fr]  # Map frame number to array index
        cx = (r.xmin + r.xmax) / 2.0
        cy = (r.ymin + r.ymax) / 2.0
        
        # Find which cell this point belongs to
        cell_idx = -1
        for i, cell in enumerate(grid_cells):
            if (cell['xmin'] <= cx < cell['xmax'] and 
                cell['ymin'] <= cy < cell['ymax']):
                cell_idx = i
                break
        
        if cell_idx == -1:
            continue  # Point outside all cells (shouldn't happen)
        
        cls = r._1  # pandas renames 'class' column to '_1' in itertuples
        if cls in class_to_idx:
            raw[fr_idx, cell_idx, class_to_idx[cls]] += 1.0
        
        area = max(0.0, (r.xmax - r.xmin) * (r.ymax - r.ymin))
        raw[fr_idx, cell_idx, 3] += area
        conf_sums[fr_idx, cell_idx] += r.confidence
        conf_counts[fr_idx, cell_idx] += 1

    # Compute mean confidence
    nonzero = conf_counts > 0
    raw[nonzero, 4] = conf_sums[nonzero] / conf_counts[nonzero]

    if normalize_by_frame_area:
        frame_area = frame_width * frame_height
        raw[..., 3] = raw[..., 3] / (frame_area + 1e-12)

    # Flatten per-frame
    embeddings = raw.reshape(n_frames, n_cells * 5)

    # Create per-frame summary
    total_counts = raw[..., 0:3].sum(axis=1)
    total_bbox_area = raw[..., 3].sum(axis=1)
    mean_conf = np.zeros(n_frames)
    total_conf_count = conf_counts.sum(axis=1)
    total_conf_sum = conf_sums.sum(axis=1)
    mean_conf[total_conf_count > 0] = total_conf_sum[total_conf_count > 0] / total_conf_count[total_conf_count > 0]

    per_frame_summary = pd.DataFrame({
        'frame': unique_frames,
        'count_adult': total_counts[:,0],
        'count_chick': total_counts[:,1],
        'count_fish' : total_counts[:,2],
        'total_bbox_area': total_bbox_area,
        'mean_confidence': mean_conf
    })

    return embeddings, np.array(unique_frames), per_frame_summary, grid_cells, raw

##############################
# Temporal smoothing
##############################

def smooth_embeddings(embeddings, window_seconds=5, fps=1):
    """
    embeddings: numpy array (n_frames, n_features)
    window_seconds * fps defines the rolling window length in frames
    Returns smoothed embeddings (same shape)
    """
    n_frames = embeddings.shape[0]
    window_frames = int(window_seconds * fps)
    if window_frames < 1:
        return embeddings.copy()

    df_emb = pd.DataFrame(embeddings)
    smoothed = df_emb.rolling(window=window_frames, min_periods=1).mean().to_numpy()
    return smoothed

##############################
# Anomaly detection (rolling Mahalanobis)
##############################

def mahalanobis_distance(x, mean, inv_cov):
    d = x - mean
    return float(np.sqrt(d.dot(inv_cov).dot(d.T)))

def compute_rolling_anomaly_scores(embeddings, window_frames=900, regularization=1e-6):
    """
    For each frame i, compute Mahalanobis distance between embeddings[i]
    and the distribution of previous `window_frames` frames (i-window .. i-1).
    Returns a float array of anomaly scores (NaN for frames with insufficient history).
    """
    n, d = embeddings.shape
    scores = np.full(n, np.nan, dtype=float)

    for i in range(n):
        start = max(0, i - window_frames)
        end = i  # exclude current
        if end - start < 2:
            # not enough history to estimate covariance reliably
            continue
        hist = embeddings[start:end]
        mean = np.mean(hist, axis=0)
        cov = np.cov(hist, rowvar=False)
        # regularize cov to avoid singularity
        cov += np.eye(d) * regularization
        try:
            inv_cov = pinv(cov)
            scores[i] = mahalanobis_distance(embeddings[i], mean, inv_cov)
        except Exception:
            scores[i] = np.nan
    return scores

##############################
# Simple plotting helpers
##############################

def plot_pca_trajectory(embeddings, frames, anomaly_scores=None, downsample=1, title="PCA trajectory"):
    """
    Reduce embeddings to 2D with PCA and plot the time trajectory.
    color by anomaly score if provided, otherwise by frame index.
    """
    idx = np.arange(0, embeddings.shape[0], downsample)
    emb_sub = embeddings[idx]
    pca = PCA(2)
    proj = pca.fit_transform(emb_sub)
    cmap = plt.get_cmap('viridis')
    plt.figure(figsize=(8,5))
    if anomaly_scores is not None:
        sc = plt.scatter(proj[:,0], proj[:,1], c=anomaly_scores[idx], cmap=cmap, s=12)
        plt.colorbar(sc, label='anomaly score')
    else:
        sc = plt.scatter(proj[:,0], proj[:,1], c=idx, cmap=cmap, s=12)
        plt.colorbar(sc, label='frame index')
    plt.title(title)
    plt.xlabel('PC1'); plt.ylabel('PC2')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_occupancy_heatmap(raw_cells, agg='mean', frame_slice=None, grid_size=None, title="Occupancy heatmap"):
    """
    raw_cells: (n_frames, grid_h, grid_w, 5)
    agg: 'mean' or 'sum'
    frame_slice: tuple(start, end) or None for all
    Plots mean count of adults across cells by default (channel 0).
    """
    if frame_slice is not None:
        start, end = frame_slice
        arr = raw_cells[start:end]
    else:
        arr = raw_cells
    if agg == 'mean':
        grid = arr[...,0].mean(axis=0)
    else:
        grid = arr[...,0].sum(axis=0)
    plt.figure(figsize=(6,5))
    plt.imshow(grid, origin='lower', aspect='auto')
    plt.colorbar(label='avg adult count')
    plt.title(title)
    plt.xlabel('grid x'); plt.ylabel('grid y')
    plt.tight_layout()
    plt.show()

def plot_time_series_summary(per_frame_summary, columns=('count_adult','count_chick','count_fish','total_bbox_area'), 
                           x_col='frame', normalize_bbox=True):
    """
    Plot time series as line graphs with option to normalize bbox area to make other metrics visible.
    x_col: 'frame' or 'time' for x-axis
    normalize_bbox: if True, plot bbox area on secondary y-axis
    """
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # Determine x-axis label and formatting
    if x_col == 'time' and 'timestamp' in per_frame_summary.columns:
        x_label = 'Time'
        # Format x-axis for datetime if we have timestamps
        import matplotlib.dates as mdates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    elif x_col == 'time':
        x_label = 'Time (seconds)'
    else:
        x_label = 'Frame'
    
    # Separate bbox area from other metrics if normalization requested
    if normalize_bbox and 'total_bbox_area' in columns:
        other_cols = [c for c in columns if c != 'total_bbox_area']
        
        # Plot count metrics on primary y-axis as lines
        for col in other_cols:
            ax1.plot(per_frame_summary[x_col], per_frame_summary[col], 
                    label=col.replace('count_', '').title(), linewidth=2, marker='o', markersize=2)
        
        ax1.set_xlabel(x_label)
        ax1.set_ylabel('Count', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot bbox area on secondary y-axis as line
        ax2 = ax1.twinx()
        ax2.plot(per_frame_summary[x_col], per_frame_summary['total_bbox_area'], 
                 color='tab:red', label='Total Bbox Area', linewidth=2, marker='s', markersize=2)
        ax2.set_ylabel('Total Bbox Area (pixels²)', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.legend(loc='upper right')
        
    else:
        # Original single-axis plot with lines
        for col in columns:
            label = col.replace('count_', '').replace('total_bbox_area', 'Total Bbox Area').title()
            ax1.plot(per_frame_summary[x_col], per_frame_summary[col], 
                    label=label, linewidth=2, marker='o', markersize=2)
        ax1.legend()
        ax1.set_xlabel(x_label)
        ax1.set_ylabel('Value')
        ax1.grid(True, alpha=0.3)
    
    plt.title('Per-frame Activity Summary')
    plt.tight_layout()
    plt.show()

def plot_pca_timeseries(embeddings, per_frame_summary, x_col='frame', n_components=3, downsample=1, title="PCA Components Over Time"):
    """
    Plot the first n_components PCA components as time series in a single figure.
    per_frame_summary: DataFrame with frame/time columns
    x_col: 'frame' or 'time' for x-axis
    """
    idx = np.arange(0, embeddings.shape[0], downsample)
    emb_sub = embeddings[idx]
    
    # Get x-axis values from the summary dataframe
    if x_col in per_frame_summary.columns:
        x_values = per_frame_summary[x_col].iloc[idx]
    else:
        print(f"Warning: Column '{x_col}' not found, using frame numbers")
        x_values = per_frame_summary['frame'].iloc[idx]
        x_col = 'frame'
    
    # Determine x-axis label and formatting
    if x_col == 'time' and 'timestamp' in per_frame_summary.columns:
        x_label = 'Time'
        use_datetime_format = True
    elif x_col == 'time':
        x_label = 'Time (seconds)'
        use_datetime_format = False
    else:
        x_label = 'Frame'
        use_datetime_format = False
    
    pca = PCA(n_components)
    proj = pca.fit_transform(emb_sub)
    
    plt.figure(figsize=(14, 8))
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    
    # Plot all components in a single figure
    for i in range(n_components):
        variance_pct = pca.explained_variance_ratio_[i] * 100
        plt.plot(x_values, proj[:, i], color=colors[i % len(colors)], 
                linewidth=2, alpha=0.8, 
                label=f'PC{i+1} ({variance_pct:.1f}% var)')
    
    plt.xlabel(x_label)
    plt.ylabel('Principal Component Value')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format x-axis for datetime if needed
    if use_datetime_format:
        import matplotlib.dates as mdates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Print explained variance
    print(f"PCA explained variance ratios: {pca.explained_variance_ratio_[:n_components]}")
    print(f"Total variance explained: {pca.explained_variance_ratio_[:n_components].sum():.1%}")

##############################
# Export helpers
##############################

def save_embeddings_csv(embeddings, frames, out_path, grid_size=(10,10), classes=('adult','chick','fish')):
    """
    Save flattened embeddings to CSV with human-friendly column names like:
      cell_r_c_count_adult, ..., cell_r_c_total_bbox_area, cell_r_c_mean_conf
    """
    grid_w, grid_h = grid_size
    cols = []
    for r in range(grid_h):
        for c in range(grid_w):
            prefix = f"cell_{r}_{c}"
            for cls in classes:
                cols.append(f"{prefix}_count_{cls}")
            cols.append(f"{prefix}_total_bbox_area")
            cols.append(f"{prefix}_mean_conf")
    df_emb = pd.DataFrame(embeddings, columns=cols)
    df_emb.insert(0, 'frame', frames)
    df_emb.to_csv(out_path, index=False)
    print(f"Saved embeddings to {out_path}")

##############################
# Example usage (put under __main__ to run)
##############################

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build frame embeddings from OD CSV")
    parser.add_argument("csv", help="path to detections CSV")
    parser.add_argument("--grid_w", type=int, default=10)
    parser.add_argument("--grid_h", type=int, default=10)
    parser.add_argument("--fps", type=int, default=25, help="frames per second (default 25 for auklab data)")
    parser.add_argument("--conf_thresh", type=float, default=0.25)
    parser.add_argument("--smooth_s", type=int, default=30, help="smoothing window in seconds")
    parser.add_argument("--nms", action='store_true', help="apply per-frame per-class NMS")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--save_emb", default=None, help="path to save embeddings CSV")
    parser.add_argument("--skip_anomaly", action='store_true', help="skip anomaly detection (faster)")
    parser.add_argument("--use_time", action='store_true', help="use time instead of frame numbers for x-axis")
    
    # Adaptive grid options
    parser.add_argument("--adaptive_grid", action='store_true', help="use adaptive grid instead of uniform grid")
    parser.add_argument("--grid_cells", type=int, default=64, help="target number of cells for adaptive grid")
    parser.add_argument("--grid_samples", type=int, default=1000, help="number of object samples for adaptive grid generation")
    parser.add_argument("--min_points", type=int, default=5, help="minimum points per cell for adaptive grid")
    parser.add_argument("--min_cell_size", type=float, default=None, help="minimum cell size (fraction of frame if ≤1, pixels if >1)")
    parser.add_argument("--max_aspect_ratio", type=float, default=None, help="maximum cell length/width ratio (e.g., 4.0 = 4:1 max)")
    parser.add_argument("--save_grid", default=None, help="path to save adaptive grid as JSON")
    parser.add_argument("--load_grid", default=None, help="path to load adaptive grid from JSON")
    args = parser.parse_args()

    print(f"Loading detections from {args.csv}...")
    df = load_detections(args.csv, conf_thresh=args.conf_thresh)
    
    # Prepare grid configuration
    adaptive_grid_config = None
    grid_cells = None
    
    if args.adaptive_grid:
        if args.load_grid:
            print(f"Loading adaptive grid from {args.load_grid}...")
            grid_cells = load_adaptive_grid(args.load_grid)
        else:
            adaptive_grid_config = {
                'target_cells': args.grid_cells,
                'n_samples': args.grid_samples,
                'min_points_per_cell': args.min_points,
                'min_cell_size': args.min_cell_size,
                'max_aspect_ratio': args.max_aspect_ratio,
                'random_seed': 42
            }
    
    print("Computing frame embeddings...")
    emb, frames, per_frame_summary, grid_cells, raw = compute_frame_embeddings(
        df,
        grid_size=(args.grid_w, args.grid_h),
        classes=('adult','chick','fish'),
        apply_nms_flag=args.nms,
        iou_thresh=args.iou,
        adaptive_grid=adaptive_grid_config,
        grid_cells=grid_cells
    )
    
    # Save adaptive grid if requested
    if args.adaptive_grid and args.save_grid and grid_cells is not None:
        save_adaptive_grid(grid_cells, args.save_grid)
        print(f"Saved adaptive grid to {args.save_grid}")
    
    # Add time column support
    if args.use_time:
        start_time = parse_start_time_from_filename(args.csv, fps=args.fps)
        per_frame_summary = add_time_column(per_frame_summary, start_time, fps=args.fps)
        x_col = 'time'
        print(f"Using time axis (FPS: {args.fps})")
        if start_time:
            print(f"Start time: {start_time}")
    else:
        x_col = 'frame'
        print("Using frame numbers for x-axis")
    
    print("Applying temporal smoothing...")
    sm = smooth_embeddings(emb, window_seconds=args.smooth_s, fps=args.fps)
    
    # Compute anomaly scores only if not skipped
    if args.skip_anomaly:
        scores = None
        print("Skipping anomaly detection (use without --skip_anomaly to enable)")
    else:
        print("Computing anomaly scores (this may take a while for large datasets)...")
        window_frames = int(args.smooth_s * args.fps)
        scores = compute_rolling_anomaly_scores(sm, window_frames=window_frames)

    print("Frames:", frames.shape, "Embedding dim:", emb.shape[1])
    
    # Generate improved plots
    print("Generating plots...")
    plot_time_series_summary(per_frame_summary, x_col=x_col, normalize_bbox=True)
    plot_pca_timeseries(sm, per_frame_summary, x_col=x_col, n_components=3)
    
    # Plot appropriate heatmap based on grid type
    if args.adaptive_grid:
        # Get frame dimensions for adaptive grid plot
        frame_width, frame_height = infer_frame_size(df)
        plot_adaptive_grid(grid_cells, frame_width, frame_height, title=f"Adaptive grid visualization ({len(grid_cells)} cells)")
    else:
        plot_occupancy_heatmap(raw, agg='mean', title=f"Average occupancy heatmap ({args.grid_w}x{args.grid_h} grid)")
    
    plot_pca_trajectory(sm, frames, anomaly_scores=scores, title="2D PCA trajectory")

    if args.save_emb:
        if args.adaptive_grid:
            # For adaptive grid, we need a different CSV format
            print("Note: Adaptive grid embeddings saved with generic column names (feature_0, feature_1, etc.)")
            df_emb = pd.DataFrame(emb, columns=[f"feature_{i}" for i in range(emb.shape[1])])
            df_emb.insert(0, 'frame', frames)
            df_emb.to_csv(args.save_emb, index=False)
            print(f"Saved adaptive embeddings to {args.save_emb}")
        else:
            save_embeddings_csv(emb, frames, args.save_emb, grid_size=(args.grid_w, args.grid_h))


# Run example with adaptive grid:
# python3 code/postprocess/post_process_od.py data/Auklab1_FAR3_2024-06-17_05.00.00_raw.csv --fps 1 --conf_thresh 0.25 --smooth_s 10 --save_emb data/embeddings_far3.csv --adaptive_grid --grid_cells 64 --grid_samples 1000 --min_points 1 --min_cell_size 20 --max_aspect_ratio 20 --save_grid data/adaptive_grid_far3.json