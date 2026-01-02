#!/usr/bin/env python3
"""
IOH Data Analysis Script (Corrected Version)
Uses JSON 'best' values which match terminal logs
Generates ECDF, ERT, and convergence plots as required by the assignment
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from pathlib import Path
from scipy import integrate

# Create output directory for plots
output_dir = Path('plots')
output_dir.mkdir(exist_ok=True)

def load_ioh_data(json_path):
    """Load data from IOH JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def extract_runs_from_json(json_data):
    """Extract run information from JSON"""
    scenario = json_data['scenarios'][0]
    runs_info = scenario['runs']

    # Extract best values for each run
    best_values = [run['best']['y'] for run in runs_info]
    best_evals = [run['best']['evals'] for run in runs_info]

    return {
        'best_values': np.array(best_values),
        'best_evals': np.array(best_evals),
        'runs_info': runs_info,
        'maximization': json_data['maximization']
    }

def load_dat_file(dat_path):
    """Load raw trajectory data from .dat file"""
    runs = []
    current_run = []

    with open(dat_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('evaluations'):
                if current_run:
                    runs.append(np.array(current_run))
                current_run = []
            elif line and not line.startswith('#'):
                parts = line.split()
                if len(parts) == 2:
                    eval_num, fitness = float(parts[0]), float(parts[1])
                    current_run.append([eval_num, fitness])

        if current_run:
            runs.append(np.array(current_run))

    return runs

def plot_convergence(runs, title, filename, maximization=True, budget=None):
    """Plot convergence curves (best-so-far fitness vs evaluations)"""
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()

    for i, run in enumerate(runs):
        if len(run) == 0:
            continue
        evals, fitness = run[:, 0], run[:, 1]

        # Compute best-so-far
        if maximization:
            best_so_far = np.maximum.accumulate(fitness)
        else:
            best_so_far = np.minimum.accumulate(fitness)

        ax.plot(evals, best_so_far, alpha=0.2, color='steelblue', linewidth=0.8)

    # Compute and plot median
    if len(runs) > 0:
        max_evals = max([run[-1, 0] for run in runs if len(run) > 0])
        if budget:
            max_evals = min(max_evals, budget)

        eval_points = np.linspace(1, max_evals, 200)
        all_best_so_far = []

        for run in runs:
            if len(run) == 0:
                continue
            evals, fitness = run[:, 0], run[:, 1]

            if maximization:
                best_so_far = np.maximum.accumulate(fitness)
            else:
                best_so_far = np.minimum.accumulate(fitness)

            interp_fitness = np.interp(eval_points, evals, best_so_far)
            all_best_so_far.append(interp_fitness)

        if all_best_so_far:
            median_curve = np.median(all_best_so_far, axis=0)
            q25 = np.percentile(all_best_so_far, 25, axis=0)
            q75 = np.percentile(all_best_so_far, 75, axis=0)

            ax.fill_between(eval_points, q25, q75, alpha=0.3, color='orange', label='IQR (25-75%)')
            ax.plot(eval_points, median_curve, 'r-', linewidth=2.5, label='Median', zorder=10)

    ax.set_xlabel('Function Evaluations', fontsize=13, fontweight='bold')
    ax.set_ylabel('Best Fitness', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filename}")

def plot_ecdf(best_values, title, filename, maximization=True):
    """Plot ECDF of final best values"""
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()

    # Sort values
    sorted_vals = np.sort(best_values)
    n = len(sorted_vals)

    if maximization:
        # For maximization: P(X >= x)
        ecdf_y = 1 - np.arange(n) / n
    else:
        # For minimization: P(X <= x)
        ecdf_y = np.arange(1, n + 1) / n

    ax.step(sorted_vals, ecdf_y, where='post', linewidth=2.5, color='steelblue')
    ax.fill_between(sorted_vals, ecdf_y, alpha=0.3, step='post', color='steelblue')

    ax.set_xlabel('Fitness Value', fontsize=13, fontweight='bold')
    if maximization:
        ax.set_ylabel('Proportion of Runs ≥ Target', fontsize=13, fontweight='bold')
    else:
        ax.set_ylabel('Proportion of Runs ≤ Target', fontsize=13, fontweight='bold')

    ax.set_title(f'ECDF: {title}', fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filename}")

def compute_ert_for_targets(runs, targets, budget, maximization=True):
    """Compute ERT for multiple targets"""
    erts = []

    for target in targets:
        successful_evals = []

        for run in runs:
            if len(run) == 0:
                continue
            evals, fitness = run[:, 0], run[:, 1]

            if maximization:
                success_mask = fitness >= target
            else:
                success_mask = fitness <= target

            if np.any(success_mask):
                first_success = np.where(success_mask)[0][0]
                successful_evals.append(evals[first_success])

        if len(successful_evals) == 0:
            erts.append(np.inf)
        else:
            unsuccessful_count = len(runs) - len(successful_evals)
            total_evals = sum(successful_evals) + (unsuccessful_count * budget)
            ert = total_evals / len(successful_evals)
            erts.append(ert)

    return np.array(erts)

def plot_ert(runs, best_values, title, filename, maximization=True, budget=5000):
    """Plot ERT curve"""
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()

    # Define target range
    if maximization:
        targets = np.linspace(np.min(best_values), np.max(best_values) * 0.99, 25)
    else:
        targets = np.linspace(np.min(best_values) * 1.01, np.max(best_values), 25)

    erts = compute_ert_for_targets(runs, targets, budget, maximization)

    # Filter out infinite ERTs
    valid_mask = np.isfinite(erts)
    valid_targets = targets[valid_mask]
    valid_erts = erts[valid_mask]

    if len(valid_targets) > 0:
        ax.semilogy(valid_targets, valid_erts, 'o-', linewidth=2.5, markersize=6,
                    color='steelblue', markerfacecolor='orange', markeredgewidth=1.5,
                    markeredgecolor='steelblue')

        ax.axhline(y=budget, color='red', linestyle='--', linewidth=2, alpha=0.7,
                   label=f'Budget = {budget}')

        ax.set_xlabel('Target Fitness', fontsize=13, fontweight='bold')
        ax.set_ylabel('ERT (log scale)', fontsize=13, fontweight='bold')
        ax.set_title(f'Expected Running Time: {title}', fontsize=15, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--', which='both')
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {filename}")
    else:
        print(f"⚠ Warning: Could not compute valid ERT for {title}")
        plt.close()

def compute_auc(runs, budget, maximization=True):
    """Compute Area Under Curve for convergence"""
    all_auc = []

    for run in runs:
        if len(run) == 0:
            continue
        evals, fitness = run[:, 0], run[:, 1]

        # Compute best-so-far
        if maximization:
            best_so_far = np.maximum.accumulate(fitness)
        else:
            best_so_far = np.minimum.accumulate(fitness)

        # Normalize evaluations to [0, 1]
        norm_evals = evals / budget

        # Compute AUC using trapezoidal rule
        auc = integrate.trapezoid(best_so_far, norm_evals)
        all_auc.append(auc)

    return np.mean(all_auc), np.std(all_auc)

def analyze_problem(name, dat_path, json_path, budget):
    """Analyze a single problem"""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    # Load data
    json_data = load_ioh_data(json_path)
    runs_data = extract_runs_from_json(json_data)
    trajectory_runs = load_dat_file(dat_path)
    maximization = runs_data['maximization']

    best_values = runs_data['best_values']
    print(f"\n📊 Performance Statistics (20 independent runs):")
    print(f"   Mean Best Fitness: {np.mean(best_values):.4f} ± {np.std(best_values):.4f}")
    print(f"   Min Best Fitness:  {np.min(best_values):.4f}")
    print(f"   Max Best Fitness:  {np.max(best_values):.4f}")

    # Compute AUC
    auc_mean, auc_std = compute_auc(trajectory_runs, budget, maximization)
    print(f"\n📈 Area Under Curve (AUC):")
    print(f"   AUC: {auc_mean:.4f} ± {auc_std:.4f}")

    # Generate plots
    safe_name = name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
    plot_convergence(trajectory_runs, f"Convergence: {name}", f"convergence_{safe_name}.pdf",
                    maximization, budget)
    plot_ecdf(best_values, name, f"ecdf_{safe_name}.pdf", maximization)
    plot_ert(trajectory_runs, best_values, name, f"ert_{safe_name}.pdf", maximization, budget)

    return {
        'mean': np.mean(best_values),
        'std': np.std(best_values),
        'min': np.min(best_values),
        'max': np.max(best_values),
        'auc_mean': auc_mean,
        'auc_std': auc_std,
        'maximization': maximization
    }

def main():
    print("\n" + "="*70)
    print("  IOH DATA ANALYSIS - EA PRACTICAL ASSIGNMENT")
    print("="*70)

    results = {}

    # Analyze F18 LABS (GA, maximization)
    results['f18'] = analyze_problem(
        "F18 LABS (GA, D=50)",
        "ioh_data/data_f18_LABS/IOHprofiler_f18_DIM50.dat",
        "ioh_data/IOHprofiler_f18_LABS.json",
        budget=5000
    )

    # Analyze F23 N-Queens (GA, maximization)
    results['f23_nqueens'] = analyze_problem(
        "F23 N-Queens (GA, D=49)",
        "ioh_data/data_f23_NQueens/IOHprofiler_f23_DIM49.dat",
        "ioh_data/IOHprofiler_f23_NQueens.json",
        budget=5000
    )

    # Analyze F23 Katsuura (ES, minimization)
    results['f23_katsuura'] = analyze_problem(
        "F23 Katsuura (ES, D=10)",
        "ioh_data/data_f23_Katsuura/IOHprofiler_f23_DIM10.dat",
        "ioh_data/IOHprofiler_f23_Katsuura.json",
        budget=50000
    )

    # Generate comprehensive summary
    print("\n" + "="*70)
    print("  COMPREHENSIVE SUMMARY")
    print("="*70)

    print("\n🔹 Part 1: Genetic Algorithm")
    print("\n  F18 LABS (D=50, 5000 evaluations):")
    print(f"    Best Fitness: {results['f18']['mean']:.4f} ± {results['f18']['std']:.4f}")
    print(f"    Range: [{results['f18']['min']:.4f}, {results['f18']['max']:.4f}]")
    print(f"    AUC: {results['f18']['auc_mean']:.2f} ± {results['f18']['auc_std']:.2f}")

    print("\n  F23 N-Queens (D=49, 5000 evaluations):")
    print(f"    Best Fitness: {results['f23_nqueens']['mean']:.4f} ± {results['f23_nqueens']['std']:.4f}")
    print(f"    Range: [{results['f23_nqueens']['min']:.4f}, {results['f23_nqueens']['max']:.4f}]")
    print(f"    AUC: {results['f23_nqueens']['auc_mean']:.2f} ± {results['f23_nqueens']['auc_std']:.2f}")

    print("\n🔹 Part 2: Evolution Strategy")
    print("\n  F23 Katsuura (D=10, 50000 evaluations):")
    print(f"    Best Fitness: {results['f23_katsuura']['mean']:.4f} ± {results['f23_katsuura']['std']:.4f}")
    print(f"    Range: [{results['f23_katsuura']['min']:.4f}, {results['f23_katsuura']['max']:.4f}]")
    print(f"    AUC: {results['f23_katsuura']['auc_mean']:.2f} ± {results['f23_katsuura']['auc_std']:.2f}")

    print(f"\n📁 All plots saved to: {output_dir.absolute()}/")
    print("\n✅ Analysis complete! Generated:")
    print("   • 3 Convergence plots (median + IQR + individual runs)")
    print("   • 3 ECDF plots (empirical cumulative distribution)")
    print("   • 3 ERT plots (expected running time)")

    # Save summary to file
    with open('analysis_summary.txt', 'w') as f:
        f.write("IOH DATA ANALYSIS SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write("Part 1: Genetic Algorithm\n\n")
        f.write(f"F18 LABS (D=50, Budget=5000):\n")
        f.write(f"  Mean ± Std: {results['f18']['mean']:.4f} ± {results['f18']['std']:.4f}\n")
        f.write(f"  AUC: {results['f18']['auc_mean']:.2f} ± {results['f18']['auc_std']:.2f}\n\n")
        f.write(f"F23 N-Queens (D=49, Budget=5000):\n")
        f.write(f"  Mean ± Std: {results['f23_nqueens']['mean']:.4f} ± {results['f23_nqueens']['std']:.4f}\n")
        f.write(f"  AUC: {results['f23_nqueens']['auc_mean']:.2f} ± {results['f23_nqueens']['auc_std']:.2f}\n\n")
        f.write("Part 2: Evolution Strategy\n\n")
        f.write(f"F23 Katsuura (D=10, Budget=50000):\n")
        f.write(f"  Mean ± Std: {results['f23_katsuura']['mean']:.4f} ± {results['f23_katsuura']['std']:.4f}\n")
        f.write(f"  AUC: {results['f23_katsuura']['auc_mean']:.2f} ± {results['f23_katsuura']['auc_std']:.2f}\n")

    print("\n💾 Summary saved to: analysis_summary.txt")

if __name__ == '__main__':
    main()
