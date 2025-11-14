# Batch ID System for Roboflow Uploads

## Problem Solved

Previously, uploading new batches with the same problem type (e.g., "fish", "edge_detection") would merge them with existing batches in Roboflow, making it confusing to track which images were uploaded when.

## Solution

Each upload session now gets a **unique timestamp-based Batch ID** that's automatically added to the batch name.

### Batch Naming Format

```
{prefix}_{batch_id}_{problem_type}
```

**Example batch names:**
- `active_learning_20251114_143022_fish`
- `active_learning_20251114_143022_edge_detection`
- `active_learning_20251115_091530_fish` (next day's upload)

### Batch ID Format

The batch ID is a timestamp: `YYYYMMDD_HHMMSS`

- `20251114_143022` = November 14, 2025 at 14:30:22
- Ensures each upload session is unique
- Groups related problem types from the same run

## Usage

### Automatic (Recommended)

The batch ID is automatically generated when you start an upload:

```bash
python upload_to_roboflow.py \
    --frames-dir data/active_learning_TRI3_batch1/frames \
    --api-key YOUR_KEY
```

Or with the simple script:

```bash
python upload_batches_simple.py data/active_learning_TRI3_batch1/frames
```

### Manual Batch ID

You can specify a custom batch ID if needed:

```bash
python upload_to_roboflow.py \
    --frames-dir data/active_learning_TRI3_batch1/frames \
    --batch-id "TRI3_june_2025" \
    --api-key YOUR_KEY
```

This will create batches like:
- `active_learning_TRI3_june_2025_fish`
- `active_learning_TRI3_june_2025_edge_detection`

## Resume Functionality

The batch ID is saved in the `.upload_state.json` file:

- **When resuming**: Uses the same batch ID from the previous attempt
- **When starting fresh**: Generates a new batch ID

```bash
# First attempt (interrupted)
python upload_batches_simple.py data/frames  # Creates batch_id: 20251114_143022

# Resume (uses same batch ID)
python upload_batches_simple.py data/frames  # Continues with: 20251114_143022

# Start fresh (new batch ID)
# When prompted, choose "n" to start new upload  # Creates batch_id: 20251114_150000
```

## Benefits

1. **No More Merging**: Each upload session creates separate batches in Roboflow
2. **Easy Tracking**: Timestamp shows when images were uploaded
3. **Grouped by Session**: All problem types from one run share the same batch ID
4. **Resume Safe**: Interrupted uploads continue with the same batch ID

## In Roboflow UI

Your batches will now appear as:

```
📁 active_learning_20251114_143022_fish (50 images)
📁 active_learning_20251114_143022_edge_detection (30 images)
📁 active_learning_20251114_143022_high_count (20 images)
📁 active_learning_20251115_091530_fish (45 images)  ← Next day
📁 active_learning_20251115_091530_edge_detection (35 images)
```

Each batch is separate and easy to find!

## Upgrade Notes

- ✅ No breaking changes - existing code works as before
- ✅ Batch IDs are automatically generated
- ✅ Resume functionality preserved
- ✅ Works with both `upload_to_roboflow.py` and `upload_batches_simple.py`
