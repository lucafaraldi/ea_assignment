#!/usr/bin/env python3
"""
IOH Data Analysis Script
Generates ECDF, ERT, and convergence plots as required by the assignment
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from pathlib import Path

# Create output directory for plots
output_dir = Path('plots')
output_dir.mkdir(exist_ok=True)

def load_ioh_data(json_path):
    """Load data from IOH JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def load_dat_file(dat_path):
    """Load raw data from .dat file"""
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

def compute_ecdf(targets_reached):
    """Compute ECDF from target achievements"""
    sorted_vals = np.sort(targets_reached)
    n = len(sorted_vals)
    ecdf_y = np.arange(1, n + 1) / n
    return sorted_vals, ecdf_y

def compute_ert(runs, target, budget, maximization=True):
    """
    Compute Expected Running Time (ERT) to reach a target
    ERT = (budget * unsuccessful_runs + sum_of_evals_for_successful) / successful_runs
    """
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
        return np.inf

    unsuccessful_count = len(runs) - len(successful_evals)
    total_evals = sum(successful_evals) + (unsuccessful_count * budget)
    ert = total_evals / len(successful_evals)

    return ert

def plot_convergence(runs, title, filename, maximization=True, budget=None):
    """Plot convergence curves (best-so-far fitness vs evaluations)"""
    plt.figure(figsize=(10, 6))

    for i, run in enumerate(runs):
        if len(run) == 0:
            continue
        evals, fitness = run[:, 0], run[:, 1]

        # Compute best-so-far
        if maximization:
            best_so_far = np.maximum.accumulate(fitness)
        else:
            best_so_far = np.minimum.accumulate(fitness)

        plt.plot(evals, best_so_far, alpha=0.3, color='blue', linewidth=0.5)

    # Compute and plot median
    if len(runs) > 0:
        # Interpolate all runs to same evaluation points
        max_evals = max([run[-1, 0] for run in runs if len(run) > 0])
        if budget:
            max_evals = min(max_evals, budget)

        eval_points = np.linspace(1, max_evals, 100)
        all_best_so_far = []

        for run in runs:
            if len(run) == 0:
                continue
            evals, fitness = run[:, 0], run[:, 1]

            if maximization:
                best_so_far = np.maximum.accumulate(fitness)
            else:
                best_so_far = np.minimum.accumulate(fitness)

            # Interpolate to eval_points
            interp_fitness = np.interp(eval_points, evals, best_so_far)
            all_best_so_far.append(interp_fitness)

        if all_best_so_far:
            median_curve = np.median(all_best_so_far, axis=0)
            plt.plot(eval_points, median_curve, 'r-', linewidth=2, label='Median')

    plt.xlabel('Function Evaluations', fontsize=12)
    plt.ylabel('Best Fitness', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def plot_ecdf_targets(runs, title, filename, maximization=True, budget=None):
    """Plot ECDF for multiple targets"""
    plt.figure(figsize=(10, 6))

    # Extract final fitness values
    final_fitness = []
    for run in runs:
        if len(run) > 0:
            final_fitness.append(run[-1, 1])

    if not final_fitness:
        print(f"Warning: No data for {title}")
        return

    final_fitness = np.array(final_fitness)

    # Define targets based on fitness range
    if maximization:
        targets = np.linspace(np.min(final_fitness), np.max(final_fitness), 20)
    else:
        targets = np.linspace(np.min(final_fitness), np.max(final_fitness), 20)

    success_rates = []
    for target in targets:
        if maximization:
            successes = np.sum(final_fitness >= target)
        else:
            successes = np.sum(final_fitness <= target)
        success_rates.append(successes / len(runs))

    plt.plot(targets, success_rates, 'b-o', linewidth=2, markersize=4)
    plt.xlabel('Target Fitness', fontsize=12)
    plt.ylabel('Proportion of Runs', fontsize=12)
    plt.title(f'ECDF: {title}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def plot_ert_curve(runs, title, filename, maximization=True, budget=None):
    """Plot ERT curve for multiple targets"""
    plt.figure(figsize=(10, 6))

    # Extract final fitness values
    final_fitness = []
    for run in runs:
        if len(run) > 0:
            final_fitness.append(run[-1, 1])

    if not final_fitness:
        print(f"Warning: No data for {title}")
        return

    final_fitness = np.array(final_fitness)

    # Define targets
    if maximization:
        targets = np.linspace(np.min(final_fitness), np.max(final_fitness) * 0.95, 15)
    else:
        targets = np.linspace(np.min(final_fitness) * 1.05, np.max(final_fitness), 15)

    erts = []
    valid_targets = []

    for target in targets:
        ert = compute_ert(runs, target, budget if budget else 50000, maximization)
        if ert != np.inf:
            erts.append(ert)
            valid_targets.append(target)

    if valid_targets:
        plt.semilogy(valid_targets, erts, 'b-o', linewidth=2, markersize=4)
        plt.xlabel('Target Fitness', fontsize=12)
        plt.ylabel('ERT (log scale)', fontsize=12)
        plt.title(f'ERT Curve: {title}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, which='both')
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
    else:
        print(f"Warning: Could not compute valid ERT for {title}")

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
        from scipy import integrate
        auc = integrate.trapezoid(best_so_far, norm_evals)
        all_auc.append(auc)

    return np.mean(all_auc), np.std(all_auc)

def analyze_problem(name, dat_path, json_path, maximization, budget):
    """Analyze a single problem"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {name}")
    print(f"{'='*60}")

    # Load data
    runs = load_dat_file(dat_path)
    json_data = load_ioh_data(json_path)

    print(f"Loaded {len(runs)} runs")

    # Compute statistics
    final_fitness = [run[-1, 1] for run in runs if len(run) > 0]
    mean_fitness = np.mean(final_fitness)
    std_fitness = np.std(final_fitness)
    min_fitness = np.min(final_fitness)
    max_fitness = np.max(final_fitness)

    print(f"\nFinal Fitness Statistics:")
    print(f"  Mean: {mean_fitness:.4f}")
    print(f"  Std:  {std_fitness:.4f}")
    print(f"  Min:  {min_fitness:.4f}")
    print(f"  Max:  {max_fitness:.4f}")

    # Compute AUC
    auc_mean, auc_std = compute_auc(runs, budget, maximization)
    print(f"\nAUC Statistics:")
    print(f"  Mean: {auc_mean:.4f}")
    print(f"  Std:  {auc_std:.4f}")

    # Generate plots
    safe_name = name.replace(' ', '_').replace('/', '_')
    plot_convergence(runs, f"Convergence: {name}", f"convergence_{safe_name}.pdf", maximization, budget)
    plot_ecdf_targets(runs, name, f"ecdf_{safe_name}.pdf", maximization, budget)
    plot_ert_curve(runs, name, f"ert_{safe_name}.pdf", maximization, budget)

    return {
        'mean': mean_fitness,
        'std': std_fitness,
        'min': min_fitness,
        'max': max_fitness,
        'auc_mean': auc_mean,
        'auc_std': auc_std
    }

def main():
    print("IOH Data Analysis")
    print("="*60)

    # Analyze F18 LABS (GA, maximization)
    f18_stats = analyze_problem(
        "F18 LABS (GA)",
        "ioh_data/data_f18_LABS/IOHprofiler_f18_DIM50.dat",
        "ioh_data/IOHprofiler_f18_LABS.json",
        maximization=True,
        budget=5000
    )

    # Analyze F23 N-Queens (GA, maximization)
    f23_nqueens_stats = analyze_problem(
        "F23 N-Queens (GA)",
        "ioh_data/data_f23_NQueens/IOHprofiler_f23_DIM49.dat",
        "ioh_data/IOHprofiler_f23_NQueens.json",
        maximization=True,
        budget=5000
    )

    # Analyze F23 Katsuura (ES, minimization)
    f23_katsuura_stats = analyze_problem(
        "F23 Katsuura (ES)",
        "ioh_data/data_f23_Katsuura/IOHprofiler_f23_DIM10.dat",
        "ioh_data/IOHprofiler_f23_Katsuura.json",
        maximization=False,
        budget=50000
    )

    # Generate summary report
    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)

    print("\nF18 LABS (D=50, GA, 5000 evals):")
    print(f"  Mean ± Std: {f18_stats['mean']:.4f} ± {f18_stats['std']:.4f}")
    print(f"  AUC: {f18_stats['auc_mean']:.2f} ± {f18_stats['auc_std']:.2f}")

    print("\nF23 N-Queens (D=49, GA, 5000 evals):")
    print(f"  Mean ± Std: {f23_nqueens_stats['mean']:.4f} ± {f23_nqueens_stats['std']:.4f}")
    print(f"  AUC: {f23_nqueens_stats['auc_mean']:.2f} ± {f23_nqueens_stats['auc_std']:.2f}")

    print("\nF23 Katsuura (D=10, ES, 50000 evals):")
    print(f"  Mean ± Std: {f23_katsuura_stats['mean']:.4f} ± {f23_katsuura_stats['std']:.4f}")
    print(f"  AUC: {f23_katsuura_stats['auc_mean']:.2f} ± {f23_katsuura_stats['auc_std']:.2f}")

    print(f"\nAll plots saved to: {output_dir}/")
    print("\nGenerated plots:")
    print("  - Convergence curves (median + individual runs)")
    print("  - ECDF curves (target achievement)")
    print("  - ERT curves (expected running time)")

if __name__ == '__main__':
    main()
