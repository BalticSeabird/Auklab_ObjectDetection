#!/usr/bin/env python3
"""
tune_parameters.py

Systematically tune detection algorithm parameters to optimize performance.
Tests different combinations and reports best settings.

Usage:
    python3 code/postprocess/tune_parameters.py
"""

import pandas as pd
import numpy as np
import sys
import os
from itertools import product

# Import benchmark functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from benchmark_event_detection import load_annotations, benchmark_algorithm, calculate_metrics


def test_parameter_combination(annotations, params, tolerance_seconds=3):
    """Test a single parameter combination and return metrics"""
    results = benchmark_algorithm(
        annotations,
        tolerance_seconds=tolerance_seconds,
        conf_thresh=params['conf_thresh'],
        smooth_window_s=params['smooth_window_s'],
        error_window_s=params['error_window_s'],
        hold_seconds=params['hold_seconds']
    )
    
    # Overall metrics
    tp = results['total_matches']
    fp = results['total_false_positives']
    fn = results['total_false_negatives']
    precision, recall, f1 = calculate_metrics(tp, fp, fn)
    
    # Per-type metrics
    arrival_tp = results['by_type']['arrival']['matches']
    arrival_fp = results['by_type']['arrival']['fp']
    arrival_fn = results['by_type']['arrival']['fn']
    arrival_prec, arrival_rec, arrival_f1 = calculate_metrics(arrival_tp, arrival_fp, arrival_fn)
    
    departure_tp = results['by_type']['departure']['matches']
    departure_fp = results['by_type']['departure']['fp']
    departure_fn = results['by_type']['departure']['fn']
    departure_prec, departure_rec, departure_f1 = calculate_metrics(departure_tp, departure_fp, departure_fn)
    
    return {
        'params': params.copy(),
        'overall': {'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn},
        'arrival': {'precision': arrival_prec, 'recall': arrival_rec, 'f1': arrival_f1, 
                   'tp': arrival_tp, 'fp': arrival_fp, 'fn': arrival_fn},
        'departure': {'precision': departure_prec, 'recall': departure_rec, 'f1': departure_f1,
                     'tp': departure_tp, 'fp': departure_fp, 'fn': departure_fn}
    }


def grid_search(annotations, param_grid, tolerance_seconds=3):
    """Perform grid search over parameter space"""
    print("\nStarting parameter grid search...")
    print(f"Testing {np.prod([len(v) for v in param_grid.values()])} combinations")
    print("This may take a few minutes...\n")
    
    results = []
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    
    for i, combination in enumerate(product(*values)):
        params = dict(zip(keys, combination))
        
        # Show progress
        if i % 10 == 0:
            print(f"Progress: {i}/{np.prod([len(v) for v in param_grid.values()])}", end='\r')
        
        result = test_parameter_combination(annotations, params, tolerance_seconds)
        results.append(result)
    
    print(f"Progress: {len(results)}/{len(results)} - Complete!")
    
    return results


def print_top_results(results, n=10, metric='f1', event_type='overall'):
    """Print top N parameter combinations by specified metric"""
    sorted_results = sorted(results, key=lambda x: x[event_type][metric], reverse=True)
    
    print(f"\n{'='*100}")
    print(f"TOP {n} PARAMETER COMBINATIONS BY {event_type.upper()} {metric.upper()}")
    print('='*100)
    
    for i, result in enumerate(sorted_results[:n], 1):
        params = result['params']
        metrics = result[event_type]
        
        print(f"\n#{i} - {metric.upper()}: {metrics[metric]:.1%}")
        print(f"   Parameters: smooth_window={params['smooth_window_s']}s, "
              f"error_window={params['error_window_s']}s, "
              f"hold={params['hold_seconds']}s, "
              f"conf={params['conf_thresh']:.2f}")
        print(f"   Overall:    Precision={result['overall']['precision']:.1%}, "
              f"Recall={result['overall']['recall']:.1%}, "
              f"F1={result['overall']['f1']:.1%} "
              f"(TP={result['overall']['tp']}, FP={result['overall']['fp']}, FN={result['overall']['fn']})")
        print(f"   Arrivals:   Precision={result['arrival']['precision']:.1%}, "
              f"Recall={result['arrival']['recall']:.1%}, "
              f"F1={result['arrival']['f1']:.1%} "
              f"(TP={result['arrival']['tp']}, FP={result['arrival']['fp']}, FN={result['arrival']['fn']})")
        print(f"   Departures: Precision={result['departure']['precision']:.1%}, "
              f"Recall={result['departure']['recall']:.1%}, "
              f"F1={result['departure']['f1']:.1%} "
              f"(TP={result['departure']['tp']}, FP={result['departure']['fp']}, FN={result['departure']['fn']})")


def main():
    # Load annotations
    print("Loading annotations...")
    annotations = load_annotations('data/Annotated_arrivals_departures.csv')
    
    event_count = sum(1 for a in annotations if a['has_events'])
    arrival_count = sum(1 for a in annotations if a['has_events'] and a['event_type'] == 'arrival')
    departure_count = sum(1 for a in annotations if a['has_events'] and a['event_type'] == 'departure')
    
    print(f"Loaded {len(annotations)} annotation entries")
    print(f"  - {event_count} events total ({arrival_count} arrivals, {departure_count} departures)")
    
    # Define parameter grid
    param_grid = {
        'conf_thresh': [0.25],  # Keep constant for now
        'smooth_window_s': [2, 3, 4, 5],  # Test different smoothing
        'error_window_s': [5, 8, 10, 12, 15],  # Test different "before" windows
        'hold_seconds': [4, 6, 8, 10, 12]  # Test different "after" windows
    }
    
    print("\nParameter ranges to test:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    # Run grid search
    results = grid_search(annotations, param_grid, tolerance_seconds=3)
    
    # Show top results by different metrics
    print_top_results(results, n=5, metric='f1', event_type='overall')
    print("\n" + "="*100)
    print_top_results(results, n=5, metric='f1', event_type='arrival')
    print("\n" + "="*100)
    print_top_results(results, n=5, metric='f1', event_type='departure')
    
    # Find best balanced result
    print("\n" + "="*100)
    print("BEST BALANCED PARAMETERS (optimizing overall F1)")
    print("="*100)
    best_overall = max(results, key=lambda x: x['overall']['f1'])
    print(f"\nRecommended parameters:")
    print(f"  smooth_window_s = {best_overall['params']['smooth_window_s']}")
    print(f"  error_window_s = {best_overall['params']['error_window_s']}")
    print(f"  hold_seconds = {best_overall['params']['hold_seconds']}")
    print(f"  conf_thresh = {best_overall['params']['conf_thresh']}")
    print(f"\nExpected performance:")
    print(f"  Overall F1: {best_overall['overall']['f1']:.1%}")
    print(f"  Arrival F1: {best_overall['arrival']['f1']:.1%}")
    print(f"  Departure F1: {best_overall['departure']['f1']:.1%}")
    
    # Save results to CSV
    results_df = []
    for r in results:
        row = {
            'smooth_window_s': r['params']['smooth_window_s'],
            'error_window_s': r['params']['error_window_s'],
            'hold_seconds': r['params']['hold_seconds'],
            'conf_thresh': r['params']['conf_thresh'],
            'overall_f1': r['overall']['f1'],
            'overall_precision': r['overall']['precision'],
            'overall_recall': r['overall']['recall'],
            'arrival_f1': r['arrival']['f1'],
            'arrival_precision': r['arrival']['precision'],
            'arrival_recall': r['arrival']['recall'],
            'departure_f1': r['departure']['f1'],
            'departure_precision': r['departure']['precision'],
            'departure_recall': r['departure']['recall']
        }
        results_df.append(row)
    
    df = pd.DataFrame(results_df)
    output_file = 'data/parameter_tuning_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✅ Full results saved to: {output_file}")


if __name__ == "__main__":
    main()
