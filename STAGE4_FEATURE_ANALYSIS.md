# Stage4 Feature Analysis & Enhancement Proposal

## Question 1: True Bird Arrival Detection

### Current Features (✓ Partial Coverage)

**What the model looks at:**
- `bird_frames` - How many frames have bird detections
- `bird_displacement` - Total distance bird moved (first to last position)
- `bird_mean_motion` - Average frame-to-frame movement
- `bird_path_efficiency` - How straight the path is vs. total distance

**What's MISSING:**
- **Number of birds**: Are there 1, 2, or 3 birds in the clip?
- **Bird count CHANGE**: Did a new bird appear? (Most important for detecting arrival!)
- **Timing of appearance**: When in the clip did the bird(s) first appear?
- **Movement timing**: Does the bird movement happen AFTER initial appearance, or before?

**The Problem:**
The current model can detect if there's movement, but NOT if it's because a **new bird arrived** vs. an existing bird already in frame just moving around.

---

## Question 2: Fish Arrival Characteristics

### Current Features (✓ Partial Coverage)

**What the model looks at:**
- `fish_detection_count` - Total number of fish bounding boxes
- `fish_presence_ratio` - % of frames with fish
- `fish_first_frame_ratio` - When fish first appears (normalized)
- `fish_last_frame_ratio` - When fish last appears
- `fish_conf_late_minus_early` - Confidence trend (improving later?)
- `fish_bird_mean_distance` - Average distance between fish and bird
- `fish_bird_min_distance` - Closest distance achieved

**What's MISSING:**

| Behavior | Current Feature | Missing Feature | Why It Matters |
|----------|-----------------|-----------------|----------------|
| **Fish appears (not there before)** | `fish_first_frame_ratio` | `fish_appeared_new` | Distinguishes "fish appears during event" from "fish was already there" |
| **Fish moves into frame** | None! | `fish_movement_distance` | Captures fish traveling to feeding spot |
| **Fish slows down** (stops at feeding spot) | None! | `fish_velocity_deceleration` | High early motion, low late motion indicates "arrived and stopped" |
| **Fish and bird converge** | `fish_bird_min_distance` | `fish_bird_convergence_rate` | Are they moving toward same location? |
| **Fish stops at bird location** | None! | `fish_bird_final_distance_ratio` | Fish ends up near where bird stops? |

---

## Proposed New Features to Add

### A. Bird Arrival Features

**1. `bird_count_early` (frames 0-33%)**
```python
bird_detections_early = bird[bird["frame"] <= total_frames * 0.33]
bird_ids_early = set(bird_detections_early["id"].unique()) if "id" in bird else {1}
bird_count_early = len(bird_ids_early)
```
**Why:** Establishes baseline number of birds before arrival

**2. `bird_count_late` (frames 66-100%)**
```python
bird_detections_late = bird[bird["frame"] >= total_frames * 0.66]
bird_ids_late = set(bird_detections_late["id"].unique()) if "id" in bird else {1}
bird_count_late = len(bird_ids_late)
```
**Why:** Captures if new birds appeared

**3. `bird_count_increased`**
```python
bird_count_increased = 1 if bird_count_late > bird_count_early else 0
```
**Why:** Direct signal: Did a new bird arrive? (Binary: 0 or 1)

**4. `bird_motion_after_appearance`**
```python
first_appearance = min(bird["frame"]) if not bird.empty else 0
motion_after = bird[bird["frame"] >= first_appearance + frames_after_appearance]
motion_after_distance = calculate_motion(motion_after)
```
**Why:** Distinguish "bird appeared then moved" from "bird just happened to move"

---

### B. Fish Arrival Features

**5. `fish_count_early` (frames 0-33%)**
```python
fish_detections_early = fish[fish["frame"] <= total_frames * 0.33]
fish_count_early = len(fish_detections_early)
```
**Why:** Was fish already in frame at start?

**6. `fish_count_late` (frames 66-100%)**
```python
fish_detections_late = fish[fish["frame"] >= total_frames * 0.66]
fish_count_late = len(fish_detections_late)
```
**Why:** Did fish appear during the event?

**7. `fish_appeared_during_event`**
```python
fish_appeared = 1 if (fish_count_early == 0 and fish_count_late > 0) else 0
```
**Why:** Direct signal: Fish wasn't there, then appeared (true arrival!)

**8. `fish_movement_distance` (how far fish traveled)**
```python
if not fish.empty:
    fish_positions_by_frame = centroids_by_frame(fish)
    if len(fish_positions_by_frame) >= 2:
        first_pos = fish_positions_by_frame[min(fish_positions_by_frame.keys())]
        last_pos = fish_positions_by_frame[max(fish_positions_by_frame.keys())]
        fish_movement_distance = ((first_pos[0]-last_pos[0])**2 + (first_pos[1]-last_pos[1])**2) ** 0.5
```
**Why:** Fish that traveled distance likely arrived at feeding spot

**9. `fish_velocity_early` vs `fish_velocity_late` → `fish_deceleration`**
```python
# Early phase motion (frames 0-50%)
early_positions = [p for f, p in fish_positions_by_frame.items() if f <= total_frames * 0.5]
early_velocity = avg_frame_to_frame_distance(early_positions)

# Late phase motion (frames 50-100%)
late_positions = [p for f, p in fish_positions_by_frame.items() if f > total_frames * 0.5]
late_velocity = avg_frame_to_frame_distance(late_positions)

fish_deceleration = early_velocity - late_velocity  # Positive = slowed down
```
**Why:** Fish "arriving" shows high motion early, stops moving late = deceleration

**10. `fish_bird_convergence_rate`**
```python
# Distance in early phase
early_overlap = sorted(set(early_bird_frames) & set(early_fish_frames))
early_distance = avg_distance(early_overlap)

# Distance in late phase
late_overlap = sorted(set(late_bird_frames) & set(late_fish_frames))
late_distance = avg_distance(late_overlap)

convergence_rate = early_distance - late_distance  # Positive = getting closer
```
**Why:** If fish and bird are converging, likely they're interacting

**11. `fish_bird_final_proximity_ratio`**
```python
# How close are they at the END?
final_distance = fish_bird_min_distance  # existing feature
frame_diagonal = (width**2 + height**2) ** 0.5  # max possible distance
final_proximity = 1 - (final_distance / frame_diagonal)  # 0-1, higher = closer
```
**Why:** If they end up close, fish likely at feeding spot with bird present

---

## Summary: What to Add

| Feature | Type | Captures | Priority |
|---------|------|----------|----------|
| `bird_count_increased` | Binary | New bird arrived | 🔴 **HIGH** |
| `fish_appeared_during_event` | Binary | Fish wasn't there, then appeared | 🔴 **HIGH** |
| `fish_deceleration` | Float | Fish moved then stopped | 🔴 **HIGH** |
| `fish_movement_distance` | Float | Fish traveled to feeding spot | 🟡 **MEDIUM** |
| `fish_bird_convergence_rate` | Float | Fish and bird moving together | 🟡 **MEDIUM** |
| `fish_bird_final_proximity_ratio` | Float | They end up near each other | 🟡 **MEDIUM** |
| `bird_motion_after_appearance` | Float | Bird moved after appearing | 🟡 **MEDIUM** |

---

## Current Data Format

✓ **Confirmed** from your detections CSVs:
```
frame,class,confidence,xmin,ymin,xmax,ymax
```

**What we have:**
- ✓ Frame-by-frame detections
- ✓ Class labels (adult, fish)
- ✓ Bounding boxes (xmin, ymin, xmax, ymax)
- ✗ NO individual tracking IDs

**Workaround**: Since there's no ID column, I can:
1. **Count detections per frame** in early vs. late phases → approximates "how many birds/fish"
2. **Use spatial clustering** to group nearby detections as same individual
3. **Track when objects appear/disappear** by analyzing frame ranges

---

## Implementation Recommendation

**Phase 1 (Easy, High Impact):**
1. ✓ Add `bird_count_early` and `bird_count_late` (count detections in early/late phases)
2. ✓ Add `fish_count_early` and `fish_count_late` (same for fish)
3. ✓ Add `bird_count_increased` (binary: late > early?)
4. ✓ Add `fish_appeared_during_event` (binary: early=0, late>0?)
5. ✓ Add `fish_deceleration` (motion early vs. late)

**Phase 2 (Feasible):**
6. ✓ Add `fish_movement_distance` (distance traveled)
7. ✓ Add `fish_bird_convergence_rate` (are distances decreasing?)
8. ✓ Add `bird_motion_after_first_appearance` (motion after bird first seen)

**Phase 3 (Advanced, optional):**
9. Implement spatial clustering to track individual birds (more complex)

---

## What's Feasible With Current Data

Your detections have: `frame, class, confidence, xmin, ymin, xmax, ymax` (no tracking IDs)

**Can implement:**
- ✓ Count detections per frame (approximates # of birds/fish)
- ✓ Detect when objects appear/disappear (new fish arrived?)
- ✓ Calculate motion by comparing centroids across frames
- ✓ Measure deceleration (early motion vs. late motion)
- ✓ Track distance changes (are they converging?)
- ✓ Identify feeding behavior (stationary after moving)

**Trade-off:** Can't perfectly distinguish "2 different birds" vs. "1 bird moved far" without spatial clustering, but:
- High confidence and large bounding box changes suggest movement of SAME individual
- Multiple detections in same frame likely = multiple individuals

---

## Summary: Ready to Implement?

I can add these 8 new features to `stage4_modeling.py`:

1. `bird_count_early` (detections in frames 0-33%)
2. `bird_count_late` (detections in frames 66-100%)
3. `bird_count_increased` (1 if late > early)
4. `fish_count_early` (detections in frames 0-33%)
5. `fish_count_late` (detections in frames 66-100%)
6. `fish_appeared_during_event` (1 if early=0 and late>0)
7. `fish_deceleration` (early motion speed - late motion speed)
8. `fish_movement_distance` (distance fish traveled)
9. `fish_bird_convergence_rate` (distance decreasing over time?)

### Would you like me to:

**Option A:** Add all Phase 1+2 features (8 features) → retrain with more behavioral signals
**Option B:** Start with Phase 1 (5 features) → test effectiveness first
**Option C:** Custom selection → which features matter most to you?

After adding features, you'd retrain the model with your new annotations, and it should capture the biological behaviors you described!
