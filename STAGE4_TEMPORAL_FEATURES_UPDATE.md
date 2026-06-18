# Temporal Features Update - Stage4 Model Enhanced

## ✅ Improved from Simple Early/Late to Dynamic Temporal Tracking

### What Changed

**Old approach (brittle):**
- `bird_count_early` (frames 0-33%) vs `bird_count_late` (frames 66-100%)
- Problem: A bird could arrive and leave → same count
- Misses: When the arrival actually happened

**New approach (temporal dynamics):**
- `bird_appears_mid_clip` → Did bird first appear between 30-70%? (1=true arrival)
- `bird_count_peak` → What's the maximum bird count reached?
- `bird_arrival_timing_ratio` → When did peak occur? (0=start, 0.5=middle, 1=end)

---

## 🎯 New Features (6 improved + 3 kept)

### Bird Arrival Detection (3 features)

| Feature | What It Captures | Why It Matters |
|---------|-----------------|----------------|
| `bird_appears_mid_clip` | Bird first appears between 30-70% of clip | **Key signal**: True arrivals happen mid-clip, not at edges |
| `bird_count_peak` | Maximum number of birds seen | Distinguishes 1 bird moving vs. 2 birds arriving |
| `bird_arrival_timing_ratio` | When did peak occur? (0-1) | Tracks temporal signature of arrival |

**Example:**
- Clip 1: Bird present at start, just moves → `appears_mid=0`, count_peak=1, timing=0.2 (start)
- Clip 2: No bird early, 1 bird middle, 1 bird end → `appears_mid=1`, count_peak=1, timing=0.5 (middle) ✓

### Fish Arrival Behavior (3 features)

| Feature | What It Captures | Why It Matters |
|---------|-----------------|----------------|
| `fish_count_increases` | Does count increase anywhere in timeline? | **Key signal**: Fish appears (0→1) or another fish arrives (1→2) |
| `fish_count_peak` | Maximum fish count reached | Captures: 1 fish, 2 fish, 3 fish arriving |
| `fish_arrival_timing_ratio` | When did peak occur? (0-1) | Tracks *when* fish arrive (feeding behavior happens mid-clip) |

**Example:**
- Clip 1: Fish there early, stays → `count_increases=0`, peak=1, timing=0.1 (not arrival)
- Clip 2: No fish early, 1 fish middle, 1 fish end → `count_increases=1`, peak=1, timing=0.6 (arrival) ✓
- Clip 3: 1 fish early, 2 fish middle → `count_increases=1`, peak=2, timing=0.4 (second fish!) ✓

### Unchanged (Still Valuable)

- `fish_deceleration` → Early motion − Late motion (positive = slowed down)
- `fish_movement_distance` → Total distance fish traveled
- `fish_bird_convergence_rate` → Are distances decreasing?

---

## 📊 How Features Work Together

### True Bird Arrival Pattern:
```
bird_appears_mid_clip = 1        (bird appeared between 30-70%)
bird_count_peak = 1 or 2          (1 bird or multiple)
bird_displacement > threshold      (moved distance)
bird_mean_motion > threshold       (had active motion)
                    ↓
        SIGNALS: Genuine arrival
```

### True Fish Arrival Pattern:
```
fish_count_increases = 1           (count went up: 0→1 or 1→2)
fish_arrival_timing_ratio = 0.5    (peak in middle of clip)
fish_deceleration > threshold      (fast early, slow late)
fish_movement_distance > 0         (traveled distance)
fish_bird_convergence_rate > 0     (got closer to bird)
                    ↓
        SIGNALS: Fish arrived and is feeding
```

### Multiple Fish Capture:
```
fish_count_increases = 1
fish_count_peak = 2                (two fish present)
fish_arrival_timing_ratio = 0.4    (second one arrived mid-clip)
valid_multiple_fish = 1            (annotated column)
                    ↓
        SIGNALS: Second fish arrival (matches your annotation!)
```

---

## 🔧 Implementation Details

### New Helper Functions

**`object_appears_mid_clip(df, total_frames, mid_start=0.3, mid_end=0.7)`**
- Checks when object first appears
- Returns 1 if appearance is in [30%-70%] range
- Captures: "Did this arrive during clip, not at start/end?"

**`get_count_timeline(df, total_frames, num_phases=3)`**
- Divides clip into 3 equal phases
- Returns [count_phase1, count_phase2, count_phase3]
- Tracks: "When do detections peak?"

**`count_increases_in_timeline(timeline)`**
- Checks if ANY adjacent phases show increase
- Returns 1 if count goes up at any point
- Captures: "Did something arrive/multiply?"

**`get_peak_timing_ratio(df, total_frames)`**
- Returns ratio (0-1) of when peak count occurs
- 0 = peak at start, 0.5 = peak mid-clip, 1 = peak at end
- Captures: Temporal signature of arrival

---

## 📝 Transition Notes

**For retraining:**
- Old models trained with old features won't work with new model
- New training script will automatically use 27 features (19 original + 8 new)
- Feature order in `MODEL_FEATURE_ORDER` matters for loading models

**Backward compatibility:**
- Old trained models: Keep using old model file
- New training: Creates new model with new features
- Not interchangeable

---

## ✓ Ready for Testing

With these temporal features, the model should better capture:

✅ **True bird arrivals:** Only flags when bird appears mid-clip (not just movement)  
✅ **Fish feeding pattern:** Arrival + deceleration + convergence (actual feeding behavior)  
✅ **Multiple fish events:** Count increases to 2+ (matches `valid_multiple_fish`)  
✅ **Reduced false positives:** Won't flag motion-only or pre-existing fish

**Next step:** Annotate 200 events and retrain with these new temporal features!
