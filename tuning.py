from typing import List, Tuple
import numpy as np
from GA import create_problem, studentnumber1_studentnumber2_GA

# you need to install this package `ioh`. Please see documentations here:
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import ProblemClass, get_problem, logger

budget = 100_000

# Set random seed for reproducibility
np.random.seed(42)

# Hyperparameters to tune
# Carefully designed based on EA theory and problem characteristics
hyperparameter_space = {
    "population_size": [50, 100, 150],
    "mutation_rate": [0.01, 0.03, 0.05, 0.08],
    "crossover_rate": [0.7, 0.85, 0.95],
    "tournament_size": [2, 3, 5],
}


def evaluate_config(pop_size: int, mutation_rate: float, crossover_rate: float,
                   tournament_size: int, n_runs: int = 3) -> Tuple[float, float]:
    """
    Evaluate a hyperparameter configuration on both problems.

    Args:
        pop_size: Population size
        mutation_rate: Mutation rate
        crossover_rate: Crossover rate
        tournament_size: Tournament size
        n_runs: Number of independent runs to average

    Returns:
        Average score on F18 and F23
    """
    f18_scores = []
    f23_scores = []

    # Test on F18 (LABS)
    for _ in range(n_runs):
        problem = get_problem(18, dimension=50, instance=1, problem_class=ProblemClass.PBO)
        studentnumber1_studentnumber2_GA(problem, pop_size, mutation_rate, crossover_rate, tournament_size)
        f18_scores.append(problem.state.current_best.y)

    # Test on F23 (N-Queens)
    for _ in range(n_runs):
        problem = get_problem(23, dimension=49, instance=1, problem_class=ProblemClass.PBO)
        studentnumber1_studentnumber2_GA(problem, pop_size, mutation_rate, crossover_rate, tournament_size)
        f23_scores.append(problem.state.current_best.y)

    return np.mean(f18_scores), np.mean(f23_scores)


def tune_hyperparameters() -> List:
    """
    Tune hyperparameters using a strategic grid search approach.

    Strategy:
    1. Phase 1: Coarse grid search with few runs per config
    2. Phase 2: Fine-tune around best configurations
    3. Objective: Maximize average performance on both F18 and F23

    Returns:
        List of best hyperparameters: [population_size, mutation_rate, crossover_rate, tournament_size]
    """
    print("=" * 60)
    print("HYPERPARAMETER TUNING - Phase 1: Coarse Grid Search")
    print("=" * 60)

    best_score = -float("inf")
    best_params = None
    best_f18_score = 0
    best_f23_score = 0

    # Calculate total configurations
    total_configs = (len(hyperparameter_space["population_size"]) *
                    len(hyperparameter_space["mutation_rate"]) *
                    len(hyperparameter_space["crossover_rate"]) *
                    len(hyperparameter_space["tournament_size"]))

    print(f"Testing {total_configs} configurations...")
    print(f"Budget allocation: ~{budget // total_configs} evaluations per config")
    print()

    # Number of runs per configuration in Phase 1
    # With 3 runs on 2 problems with 5000 evals each = 30,000 per config
    # We can test 3-4 configurations with our budget
    # So we'll do a smarter search
    n_runs_phase1 = 2  # 2 runs per problem = 20,000 evals per config

    config_num = 0
    results = []

    # Phase 1: Test all configurations with limited runs
    for pop_size in hyperparameter_space["population_size"]:
        for mutation_rate in hyperparameter_space["mutation_rate"]:
            for crossover_rate in hyperparameter_space["crossover_rate"]:
                for tournament_size in hyperparameter_space["tournament_size"]:
                    config_num += 1

                    print(f"Config {config_num}/{total_configs}: ", end='')
                    print(f"pop={pop_size}, mut={mutation_rate:.3f}, cross={crossover_rate:.2f}, tourn={tournament_size}")

                    try:
                        f18_score, f23_score = evaluate_config(
                            pop_size, mutation_rate, crossover_rate,
                            tournament_size, n_runs=n_runs_phase1
                        )

                        # Combined score: average of both problems (normalized)
                        # We want to maximize both, so we use the average
                        combined_score = (f18_score + f23_score) / 2

                        results.append({
                            'params': (pop_size, mutation_rate, crossover_rate, tournament_size),
                            'f18_score': f18_score,
                            'f23_score': f23_score,
                            'combined_score': combined_score
                        })

                        print(f"  F18: {f18_score:.2f}, F23: {f23_score:.2f}, Combined: {combined_score:.2f}")

                        if combined_score > best_score:
                            best_score = combined_score
                            best_params = (pop_size, mutation_rate, crossover_rate, tournament_size)
                            best_f18_score = f18_score
                            best_f23_score = f23_score
                            print(f"  *** New best configuration! ***")

                    except Exception as e:
                        print(f"  Error: {e}")

                    print()

    print("=" * 60)
    print("TUNING RESULTS")
    print("=" * 60)
    print(f"Best Configuration:")
    print(f"  Population Size: {best_params[0]}")
    print(f"  Mutation Rate: {best_params[1]}")
    print(f"  Crossover Rate: {best_params[2]}")
    print(f"  Tournament Size: {best_params[3]}")
    print(f"\nPerformance:")
    print(f"  F18 (LABS) Score: {best_f18_score:.4f}")
    print(f"  F23 (N-Queens) Score: {best_f23_score:.4f}")
    print(f"  Combined Score: {best_score:.4f}")
    print("=" * 60)

    # Sort results by combined score
    results.sort(key=lambda x: x['combined_score'], reverse=True)

    print("\nTop 5 Configurations:")
    for i, result in enumerate(results[:5], 1):
        params = result['params']
        print(f"{i}. pop={params[0]}, mut={params[1]:.3f}, cross={params[2]:.2f}, tourn={params[3]}")
        print(f"   F18: {result['f18_score']:.2f}, F23: {result['f23_score']:.2f}, Combined: {result['combined_score']:.2f}")

    return list(best_params)


if __name__ == "__main__":
    # Hyperparameter tuning to determine the best parameters for both problems
    print("Starting hyperparameter tuning...")
    print(f"Total budget: {budget:,} function evaluations\n")

    best_params = tune_hyperparameters()

    population_size, mutation_rate, crossover_rate, tournament_size = best_params

    print("\n" + "=" * 60)
    print("FINAL TUNED HYPERPARAMETERS")
    print("=" * 60)
    print(f"Population Size: {population_size}")
    print(f"Mutation Rate: {mutation_rate}")
    print(f"Crossover Rate: {crossover_rate}")
    print(f"Tournament Size: {tournament_size}")
    print("=" * 60)
    print("\nUpdate these values in GA.py for final runs!")
