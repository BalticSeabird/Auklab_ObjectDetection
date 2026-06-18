# Architecture Question: One Classifier vs Two

## Current Architecture (What We Have Now)

**Single Model:**
```
Detection CSV
    ↓
Extract 27 features (bird + fish + spatial)
    ↓
Logistic Regression Model
    ↓
Output: is_new_fish_arrival (binary: bird arrival WITH fish?)
```

**Rules applied BEFORE model:**
1. Rule check: Is there a true bird arrival? (bird_appears_mid_clip=1, movement>threshold)
2. If YES → Rule check: Is there fish? (fish_count_increases=1, early motion>threshold)
3. If both YES → Run model as final decision

---

## Why Two Separate Models Would Be Better

### Problem with Current Single Model

The single model tries to answer TWO distinct questions at once:

1. **"Is there a real bird arrival?"** (behavioral: did NEW bird appear, not just movement)
2. **"Is there fish present/arriving?"** (spatial+temporal: fish appeared, moved to spot, stopped)

These require different features:

**Bird arrival:**
- `bird_appears_mid_clip` (when did it appear?)
- `bird_displacement` (how far did it move?)
- `bird_mean_motion` (active motion during arrival?)

**Fish arrival:**
- `fish_count_increases` (count went up?)
- `fish_deceleration` (moved then stopped?)
- `fish_movement_distance` (traveled distance?)
- `fish_bird_convergence_rate` (interacting with bird?)

The model tries to weight ALL of these together, but:
- `bird_displacement` matters a lot for Q1, less for Q2
- `fish_deceleration` matters a lot for Q2, nothing for Q1
- Model has to figure out which features to emphasize → harder to train

### Benefits of Two-Model Architecture

```
Model 1: ARRIVAL DETECTOR
  Input: bird detection CSV
  Features: bird_appears_mid_clip, bird_displacement, bird_mean_motion, etc.
  Output: is_true_arrival (0 or 1)
  
Model 2: FISH DETECTOR (only if is_true_arrival=1)
  Input: fish detection CSV + arrival confirmation
  Features: fish_count_increases, fish_deceleration, fish_movement_distance, etc.
  Output: is_fish_present (0 or 1)

Final: is_new_fish_arrival = is_true_arrival AND is_fish_present
```

**Advantages:**
- ✓ Each model focused on single question (easier to train, debug, interpret)
- ✓ Bird model: 75%+ accuracy easily achievable (simple question)
- ✓ Fish model: Can use high-quality bird arrival data as input (perfect upstream)
- ✓ Separates concerns (independent development, testing, iteration)
- ✓ Can improve fish model without touching bird model (and vice versa)

**Disadvantages:**
- ✗ Two models instead of one (more complexity)
- ✗ Errors cascade (false negative in bird model → miss all fish in that event)
- ✗ Training pipeline more complex (need separate validation sets)
- ✗ Only slight accuracy improvement (~2-5%) likely

---

## Recommendation

### Now (Recommended)
Stick with current **single model** because:
1. ✓ Simpler implementation
2. ✓ Easier to debug (one model, one set of features)
3. ✓ Temporal features already capture the key patterns
4. ✓ Good enough accuracy for WWF use case (~75-82%)
5. ✓ Can always split later if needed

### Test Single Model First

With the new temporal features, the single model should work well:
- `bird_appears_mid_clip` = strong signal for true arrival
- `fish_count_increases` = strong signal for new fish
- Combined with deceleration + convergence = solid classification

**After testing:** If accuracy is >80% and false positive rate is low, stop here.

---

## Later (If Needed)

Consider two-model architecture **if:**
- Single model plateaus below 75% accuracy
- False positive rate is still too high for WWF
- Need to improve specific station (e.g., ROST2-4) without affecting others
- Want interpretability (each model is simpler to explain)

**How to switch to two models:**
```python
# Model 1: Train on "is this a real arrival?"
# Use only: bird_appears_mid_clip, bird_displacement, bird_mean_motion, 
#           total_frames, bird_frames

# Model 2: Train on "is there fish?" (given real arrival)
# Use only: fish_count_increases, fish_count_peak, fish_deceleration,
#           fish_movement_distance, fish_bird_convergence_rate

# Pipeline:
# 1. Run Model 1 → is_true_arrival
# 2. If is_true_arrival=1 → Run Model 2 → is_fish_present
# 3. Final: is_new_fish_arrival = is_true_arrival AND is_fish_present
```

---

## Decision Matrix

```
Current accuracy    Fix Strategy
─────────────────────────────────────────────────────
> 80%               ✓ DONE! Ship with single model
                    Keep it simple

75-80%              ✓ GOOD ENOUGH
                    Single model is sufficient
                    
< 75%               ❓ INVESTIGATE
                    - Check if ROST2-4 is the issue
                    - Try two-model approach as next step
                    - Or collect more/better annotations
```

---

## Conclusion

**For now: Use single model** with new temporal features.

**Test and measure:** If Phase 1/2 testing shows good accuracy (>78%), you're done.

**Later: Consider two models** if single model hits a ceiling and you need further improvement.

The new temporal features are powerful enough that a single well-trained model should perform well!
