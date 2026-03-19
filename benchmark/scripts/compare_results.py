#!/usr/bin/env python3
"""Compare benchmark results across phases and backends."""
import argparse
import json
import os
import sys


def load_results(path):
    """Load results from JSON file."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    return data


def compare_two(baseline_path, current_path, regression_threshold=0.05):
    """Compare current results against baseline."""
    baseline = load_results(baseline_path)
    current = load_results(current_path)

    # Index by (model, backend)
    base_idx = {}
    for r in baseline:
        key = (r.get('model', ''), r.get('backend', ''))
        base_idx[key] = r

    regressions = []
    improvements = []

    print(f"{'Model':<30} {'Backend':<8} {'Base ms':>8} {'Curr ms':>8} "
          f"{'Change':>8} {'IOCTLs':>10}")
    print("-" * 80)

    for r in current:
        key = (r.get('model', ''), r.get('backend', ''))
        if key not in base_idx:
            print(f"{key[0]:<30} {key[1]:<8} {'N/A':>8} "
                  f"{r['avg_ms']:>8.2f}    (new)")
            continue

        b = base_idx[key]
        base_ms = b['avg_ms']
        curr_ms = r['avg_ms']
        change_pct = (curr_ms - base_ms) / base_ms

        base_ioctls = b.get('ioctl_count', '')
        curr_ioctls = r.get('ioctl_count', '')
        ioctl_str = f"{base_ioctls}→{curr_ioctls}" if curr_ioctls else str(base_ioctls)

        status = ''
        if change_pct > regression_threshold:
            status = ' REGRESSION'
            regressions.append((key, change_pct))
        elif change_pct < -0.01:
            status = ' improved'
            improvements.append((key, change_pct))

        print(f"{key[0]:<30} {key[1]:<8} {base_ms:>8.2f} {curr_ms:>8.2f} "
              f"{change_pct:>+7.1%} {ioctl_str:>10}{status}")

    print()
    if improvements:
        print(f"Improvements: {len(improvements)}")
        for (model, backend), pct in improvements:
            print(f"  {model}/{backend}: {pct:+.1%}")

    if regressions:
        print(f"\nREGRESSIONS (>{regression_threshold:.0%} slower): {len(regressions)}")
        for (model, backend), pct in regressions:
            print(f"  {model}/{backend}: {pct:+.1%}")
        return False

    print("No regressions detected.")
    return True


def summary_table(results_dir):
    """Print summary table from all JSON files in a directory."""
    all_results = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith('.json') and fname != 'system_info.json':
            all_results.extend(load_results(os.path.join(results_dir, fname)))

    if not all_results:
        print("No results found.")
        return

    # Group by model
    by_model = {}
    for r in all_results:
        model = r.get('model', 'unknown')
        by_model.setdefault(model, []).append(r)

    print(f"{'Model':<30} {'Backend':<8} {'Avg ms':>8} {'Min ms':>8} "
          f"{'P95 ms':>8} {'FPS':>8} {'IOCTLs':>8}")
    print("=" * 85)

    for model in sorted(by_model.keys()):
        for r in sorted(by_model[model], key=lambda x: x.get('backend', '')):
            backend = r.get('backend', '?')
            ioctls = r.get('ioctl_count', '')
            print(f"{model:<30} {backend:<8} {r['avg_ms']:>8.2f} "
                  f"{r['min_ms']:>8.2f} {r.get('p95_ms', 0):>8.2f} "
                  f"{r['fps']:>8.1f} {str(ioctls):>8}")
        print()


def main():
    parser = argparse.ArgumentParser(description='Compare benchmark results')
    sub = parser.add_subparsers(dest='cmd')

    cmp = sub.add_parser('compare', help='Compare baseline vs current')
    cmp.add_argument('baseline', help='Baseline results JSON')
    cmp.add_argument('current', help='Current results JSON')
    cmp.add_argument('--threshold', type=float, default=0.05,
                     help='Regression threshold (default: 5%%)')

    tbl = sub.add_parser('summary', help='Summary table from results dir')
    tbl.add_argument('results_dir', help='Directory with result JSON files')

    args = parser.parse_args()

    if args.cmd == 'compare':
        ok = compare_two(args.baseline, args.current, args.threshold)
        sys.exit(0 if ok else 1)
    elif args.cmd == 'summary':
        summary_table(args.results_dir)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
