"""
Simplified parameter tuning for state-based event detector.
Tests multiple parameter combinations to find optimal configuration.
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from benchmark_state_based_detector import benchmark_state_based
from benchmark_event_detection import load_annotations, calculate_metrics
from itertools import product

def run_tuning_experiment(annotations):
    """
    Test different parameter combinations systematically.
    """
    
    # Parameter ranges to test
    param_grid = {
        'min_stable_duration': [3, 5, 7, 10, 15],  # Minimum duration for stable period
        'max_spike_duration': [2, 3, 4],           # Spike removal threshold
        'max_dip_duration': [3, 4, 5],             # Dip filling threshold
        'min_count_change': [1, 2],                # Minimum count change to be an event
    }
    
    # Fixed parameters
    fixed_params = {
        'edge_margin': 100,
        'max_stable_variation': 0,  # Must be 0 for proper state detection
        'max_transition_duration': 15,
        'tolerance_seconds': 3,
    }
    
    print("="*80)
    print("STATE-BASED DETECTOR PARAMETER TUNING")
    print("="*80)
    print(f"\nTesting parameter combinations:")
    print(f"  min_stable_duration: {param_grid['min_stable_duration']}")
    print(f"  max_spike_duration: {param_grid['max_spike_duration']}")
    print(f"  max_dip_duration: {param_grid['max_dip_duration']}")
    print(f"  min_count_change: {param_grid['min_count_change']}")
    print(f"\nFixed parameters:")
    for k, v in fixed_params.items():
        print(f"  {k}: {v}")
    
    # Calculate total combinations
    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)
    
    print(f"\nTotal combinations to test: {total_combinations}")
    print("="*80)
    
    results = []
    
    # Test all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    for i, combo in enumerate(product(*param_values), 1):
        params = dict(zip(param_names, combo))
        params.update(fixed_params)
        
        print(f"\n[{i}/{total_combinations}] Testing: stable={params['min_stable_duration']}s, "
              f"spike={params['max_spike_duration']}s, dip={params['max_dip_duration']}s, "
              f"change={params['min_count_change']}")
        
        # Run benchmark with these parameters
        benchmark_results = benchmark_state_based(
            annotations,
            tolerance_seconds=params['tolerance_seconds'],
            edge_margin=params['edge_margin'],
            max_spike_duration=params['max_spike_duration'],
            max_dip_duration=params['max_dip_duration'],
            min_stable_duration=params['min_stable_duration'],
            max_stable_variation=params['max_stable_variation'],
            min_count_change=params['min_count_change'],
            max_transition_duration=params['max_transition_duration']
        )
        
        # Calculate metrics
        tp = benchmark_results['total_matches']
        fp = benchmark_results['total_false_positives']
        fn = benchmark_results['total_false_negatives']
        
        precision, recall, f1 = calculate_metrics(tp, fp, fn)
        
        # Calculate metrics by type
        arrival_tp = benchmark_results['by_type']['arrival']['matches']
        arrival_fp = benchmark_results['by_type']['arrival']['fp']
        arrival_fn = benchmark_results['by_type']['arrival']['fn']
        arrival_prec, arrival_rec, arrival_f1 = calculate_metrics(arrival_tp, arrival_fp, arrival_fn)
        
        departure_tp = benchmark_results['by_type']['departure']['matches']
        departure_fp = benchmark_results['by_type']['departure']['fp']
        departure_fn = benchmark_results['by_type']['departure']['fn']
        departure_prec, departure_rec, departure_f1 = calculate_metrics(departure_tp, departure_fp, departure_fn)
        
        print(f"  → F1={f1*100:.1f}% (P={precision*100:.1f}%, R={recall*100:.1f}%) | "
              f"Arr={arrival_f1*100:.1f}%, Dep={departure_f1*100:.1f}% | "
              f"TP={tp}, FP={fp}, FN={fn}")
        
        # Store results
        results.append({
            'min_stable_duration': params['min_stable_duration'],
            'max_spike_duration': params['max_spike_duration'],
            'max_dip_duration': params['max_dip_duration'],
            'min_count_change': params['min_count_change'],
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'arrival_f1': arrival_f1,
            'departure_f1': departure_f1,
        })
    
    return results


def analyze_results(results):
    """Analyze tuning results and identify best configurations"""
    
    print("\n" + "="*80)
    print("TUNING RESULTS SUMMARY")
    print("="*80)
    
    # Sort by overall F1
    results_sorted = sorted(results, key=lambda x: x['f1'], reverse=True)
    
    print("\n🏆 TOP 10 CONFIGURATIONS (by overall F1-score):")
    print("-"*80)
    
    for rank, result in enumerate(results_sorted[:10], 1):
        print(f"\n#{rank}. F1={result['f1']*100:.1f}% (Precision={result['precision']*100:.1f}%, Recall={result['recall']*100:.1f}%)")
        print(f"    Parameters: stable={result['min_stable_duration']}s, "
              f"spike={result['max_spike_duration']}s, dip={result['max_dip_duration']}s, "
              f"change={result['min_count_change']}")
        print(f"    Arrivals:   F1={result['arrival_f1']*100:.1f}%")
        print(f"    Departures: F1={result['departure_f1']*100:.1f}%")
        print(f"    Counts: {result['tp']} TP, {result['fp']} FP, {result['fn']} FN")
    
    # Find best for different criteria
    print("\n" + "="*80)
    print("BEST CONFIGURATIONS BY DIFFERENT CRITERIA:")
    print("="*80)
    
    best_f1 = max(results, key=lambda x: x['f1'])
    print(f"\n📊 Best Overall F1: {best_f1['f1']*100:.1f}%")
    print(f"   Parameters: stable={best_f1['min_stable_duration']}s, "
          f"spike={best_f1['max_spike_duration']}s, dip={best_f1['max_dip_duration']}s, "
          f"change={best_f1['min_count_change']}")
    
    best_precision = max(results, key=lambda x: x['precision'])
    print(f"\n📊 Best Precision (fewer false positives): {best_precision['precision']*100:.1f}%")
    print(f"   Parameters: stable={best_precision['min_stable_duration']}s, "
          f"spike={best_precision['max_spike_duration']}s, dip={best_precision['max_dip_duration']}s, "
          f"change={best_precision['min_count_change']}")
    print(f"   F1={best_precision['f1']*100:.1f}%, FP={best_precision['fp']}, FN={best_precision['fn']}")
    
    best_recall = max(results, key=lambda x: x['recall'])
    print(f"\n📊 Best Recall (catch most events): {best_recall['recall']*100:.1f}%")
    print(f"   Parameters: stable={best_recall['min_stable_duration']}s, "
          f"spike={best_recall['max_spike_duration']}s, dip={best_recall['max_dip_duration']}s, "
          f"change={best_recall['min_count_change']}")
    print(f"   F1={best_recall['f1']*100:.1f}%, FP={best_recall['fp']}, FN={best_recall['fn']}")
    
    best_departure = max(results, key=lambda x: x['departure_f1'])
    print(f"\n📊 Best Departure F1: {best_departure['departure_f1']*100:.1f}%")
    print(f"   Parameters: stable={best_departure['min_stable_duration']}s, "
          f"spike={best_departure['max_spike_duration']}s, dip={best_departure['max_dip_duration']}s, "
          f"change={best_departure['min_count_change']}")
    print(f"   Overall F1={best_departure['f1']*100:.1f}%")
    
    # Weighted score (prioritize departure performance)
    for r in results:
        r['weighted_f1'] = 0.6 * r['f1'] + 0.4 * r['departure_f1']
    
    best_weighted = max(results, key=lambda x: x['weighted_f1'])
    print(f"\n⚖️  Best Balanced (60% overall F1 + 40% departure F1): {best_weighted['weighted_f1']*100:.1f}%")
    print(f"   Parameters: stable={best_weighted['min_stable_duration']}s, "
          f"spike={best_weighted['max_spike_duration']}s, dip={best_weighted['max_dip_duration']}s, "
          f"change={best_weighted['min_count_change']}")
    print(f"   Overall F1={best_weighted['f1']*100:.1f}%, Departure F1={best_weighted['departure_f1']*100:.1f}%")
    print(f"   TP={best_weighted['tp']}, FP={best_weighted['fp']}, FN={best_weighted['fn']}")
    
    return results_sorted, best_weighted


def save_results(results, filename='state_based_tuning_results.txt'):
    """Save results to text file"""
    output_path = f'data/{filename}'
    
    with open(output_path, 'w') as f:
        f.write("State-Based Event Detector Parameter Tuning Results\n")
        f.write("="*80 + "\n\n")
        
        f.write("Configuration,F1,Precision,Recall,Arrival_F1,Departure_F1,TP,FP,FN\n")
        
        for r in sorted(results, key=lambda x: x['f1'], reverse=True):
            config = f"stable={r['min_stable_duration']},spike={r['max_spike_duration']},dip={r['max_dip_duration']},change={r['min_count_change']}"
            f.write(f"{config},{r['f1']:.3f},{r['precision']:.3f},{r['recall']:.3f},"
                   f"{r['arrival_f1']:.3f},{r['departure_f1']:.3f},"
                   f"{r['tp']},{r['fp']},{r['fn']}\n")
    
    print(f"\n✅ Saved detailed results to: {output_path}")


def main():
    # Load annotations
    print("Loading annotations...")
    annotations = load_annotations('data/Annotated_arrivals_departures.csv')
    print(f"Loaded {len(annotations)} annotation entries")
    
    # Count events
    event_count = sum(1 for a in annotations if a['has_events'])
    arrival_count = sum(1 for a in annotations if a['has_events'] and a['event_type'] == 'arrival')
    departure_count = sum(1 for a in annotations if a['has_events'] and a['event_type'] == 'departure')
    
    print(f"  - {event_count} events total ({arrival_count} arrivals, {departure_count} departures)")
    
    # Run tuning
    results = run_tuning_experiment(annotations)
    
    # Analyze results
    results_sorted, best_config = analyze_results(results)
    
    # Save results
    save_results(results)
    
    print("\n" + "="*80)
    print("🎯 RECOMMENDED CONFIGURATION FOR PRODUCTION:")
    print("="*80)
    print(f"\n  min_stable_duration = {best_config['min_stable_duration']}")
    print(f"  max_spike_duration = {best_config['max_spike_duration']}")
    print(f"  max_dip_duration = {best_config['max_dip_duration']}")
    print(f"  min_count_change = {best_config['min_count_change']}")
    print(f"  max_stable_variation = 0  (fixed)")
    print(f"  edge_margin = 100  (fixed)")
    print(f"\n  Expected Performance:")
    print(f"    Overall F1: {best_config['f1']*100:.1f}%")
    print(f"    Precision: {best_config['precision']*100:.1f}%")
    print(f"    Recall: {best_config['recall']*100:.1f}%")
    print(f"    Departure F1: {best_config['departure_f1']*100:.1f}%")
    
    print("\n" + "="*80)
    print("TUNING COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
