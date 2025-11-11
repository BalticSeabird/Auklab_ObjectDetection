#!/usr/bin/env python3
"""
compare_edge_filtering.py

Compare benchmark results WITH and WITHOUT edge filtering to measure the impact.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from benchmark_event_detection import load_annotations, benchmark_algorithm, print_results


def main():
    # Load annotations
    print("Loading annotations...")
    annotations = load_annotations('data/Annotated_arrivals_departures.csv')
    print(f"Loaded {len(annotations)} annotation entries")
    
    # Count annotated events
    event_count = sum(1 for a in annotations if a['has_events'])
    arrival_count = sum(1 for a in annotations if a['has_events'] and a['event_type'] == 'arrival')
    departure_count = sum(1 for a in annotations if a['has_events'] and a['event_type'] == 'departure')
    
    print(f"  - {event_count} events total")
    print(f"  - {arrival_count} arrivals")
    print(f"  - {departure_count} departures")
    
    # Run WITHOUT edge filtering
    print("\n" + "="*80)
    print("BASELINE: WITHOUT Edge Filtering")
    print("="*80)
    print("Parameters: smooth_window_s=3, error_window_s=10, hold_seconds=8")
    print("Edge filtering: DISABLED")
    
    results_no_filter = benchmark_algorithm(
        annotations,
        tolerance_seconds=3,
        conf_thresh=0.25,
        smooth_window_s=3,
        error_window_s=10,
        hold_seconds=8,
        filter_edges=False  # DISABLED
    )
    
    print_results(results_no_filter, tolerance_seconds=3)
    
    # Run WITH edge filtering
    print("\n" + "="*80)
    print("WITH Edge Filtering (margin=50px from 2688x1520 frame)")
    print("="*80)
    print("Parameters: smooth_window_s=3, error_window_s=10, hold_seconds=8")
    print("Edge filtering: ENABLED (margin=50px)")
    
    results_with_filter = benchmark_algorithm(
        annotations,
        tolerance_seconds=3,
        conf_thresh=0.25,
        smooth_window_s=3,
        error_window_s=10,
        hold_seconds=8,
        filter_edges=True,  # ENABLED
        frame_width=2688,
        frame_height=1520,
        edge_margin=50
    )
    
    print_results(results_with_filter, tolerance_seconds=3)
    
    # Calculate improvements
    print("\n" + "="*80)
    print("IMPROVEMENT ANALYSIS")
    print("="*80)
    
    # Overall metrics
    def calc_metrics(results):
        tp = results['total_matches']
        fp = results['total_false_positives']
        fn = results['total_false_negatives']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1}
    
    baseline = calc_metrics(results_no_filter)
    filtered = calc_metrics(results_with_filter)
    
    print("\n📊 OVERALL METRICS:")
    print(f"   Precision: {baseline['precision']*100:.1f}% → {filtered['precision']*100:.1f}% "
          f"({filtered['precision']*100 - baseline['precision']*100:+.1f} pp)")
    print(f"   Recall:    {baseline['recall']*100:.1f}% → {filtered['recall']*100:.1f}% "
          f"({filtered['recall']*100 - baseline['recall']*100:+.1f} pp)")
    print(f"   F1-Score:  {baseline['f1']*100:.1f}% → {filtered['f1']*100:.1f}% "
          f"({filtered['f1']*100 - baseline['f1']*100:+.1f} pp)")
    print(f"   False Positives: {baseline['fp']} → {filtered['fp']} "
          f"({filtered['fp'] - baseline['fp']:+d})")
    print(f"   False Negatives: {baseline['fn']} → {filtered['fn']} "
          f"({filtered['fn'] - baseline['fn']:+d})")
    
    # By type
    print("\n📈 ARRIVALS:")
    baseline_arr = results_no_filter['by_type']['arrival']
    filtered_arr = results_with_filter['by_type']['arrival']
    
    def calc_type_metrics(type_results):
        tp = type_results['matches']
        fp = type_results['fp']
        fn = type_results['fn']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    baseline_arr_m = calc_type_metrics(baseline_arr)
    filtered_arr_m = calc_type_metrics(filtered_arr)
    
    print(f"   Precision: {baseline_arr_m['precision']*100:.1f}% → {filtered_arr_m['precision']*100:.1f}% "
          f"({filtered_arr_m['precision']*100 - baseline_arr_m['precision']*100:+.1f} pp)")
    print(f"   Recall:    {baseline_arr_m['recall']*100:.1f}% → {filtered_arr_m['recall']*100:.1f}% "
          f"({filtered_arr_m['recall']*100 - baseline_arr_m['recall']*100:+.1f} pp)")
    print(f"   F1-Score:  {baseline_arr_m['f1']*100:.1f}% → {filtered_arr_m['f1']*100:.1f}% "
          f"({filtered_arr_m['f1']*100 - baseline_arr_m['f1']*100:+.1f} pp)")
    
    print("\n📉 DEPARTURES:")
    baseline_dep = results_no_filter['by_type']['departure']
    filtered_dep = results_with_filter['by_type']['departure']
    
    baseline_dep_m = calc_type_metrics(baseline_dep)
    filtered_dep_m = calc_type_metrics(filtered_dep)
    
    print(f"   Precision: {baseline_dep_m['precision']*100:.1f}% → {filtered_dep_m['precision']*100:.1f}% "
          f"({filtered_dep_m['precision']*100 - baseline_dep_m['precision']*100:+.1f} pp)")
    print(f"   Recall:    {baseline_dep_m['recall']*100:.1f}% → {filtered_dep_m['recall']*100:.1f}% "
          f"({filtered_dep_m['recall']*100 - baseline_dep_m['recall']*100:+.1f} pp)")
    print(f"   F1-Score:  {baseline_dep_m['f1']*100:.1f}% → {filtered_dep_m['f1']*100:.1f}% "
          f"({filtered_dep_m['f1']*100 - baseline_dep_m['f1']*100:+.1f} pp)")
    
    print("\n✅ CONCLUSION:")
    if filtered['f1'] > baseline['f1']:
        improvement = filtered['f1'] - baseline['f1']
        print(f"   Edge filtering IMPROVED overall F1-score by {improvement*100:.1f} percentage points!")
    elif filtered['f1'] < baseline['f1']:
        degradation = baseline['f1'] - filtered['f1']
        print(f"   Edge filtering DECREASED overall F1-score by {degradation*100:.1f} percentage points.")
    else:
        print("   Edge filtering had NO NET EFFECT on F1-score.")
    
    # Explain the mechanism
    if filtered['fp'] < baseline['fp']:
        print(f"   • Reduced false positives by {baseline['fp'] - filtered['fp']} (flickering edge detections)")
    if filtered['fn'] < baseline['fn']:
        print(f"   • Reduced false negatives by {baseline['fn'] - filtered['fn']} (cleaner signal)")
    if filtered['fn'] > baseline['fn']:
        print(f"   • Increased false negatives by {filtered['fn'] - baseline['fn']} (may have removed valid detections)")


if __name__ == "__main__":
    main()
