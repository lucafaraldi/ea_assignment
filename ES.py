import numpy as np
# you need to install this package `ioh`. Please see documentations here:
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass

budget = 50000
dimension = 10

# Set random seed for reproducibility
np.random.seed(42)

# ES Hyperparameters
MU = 10  # Number of parents
LAMBDA = 70  # Number of offspring (7*mu is a good rule of thumb)
INITIAL_SIGMA = 0.5  # Initial step size


def initialize_population(mu: int, dimension: int, lower_bound: float = -5.0, upper_bound: float = 5.0):
    """
    Initialize population with positions and step sizes.

    Args:
        mu: Population size
        dimension: Problem dimension
        lower_bound: Lower bound of search space
        upper_bound: Upper bound of search space

    Returns:
        Tuple of (positions, step_sizes)
    """
    # Initialize positions uniformly in search space
    positions = np.random.uniform(lower_bound, upper_bound, (mu, dimension))

    # Initialize step sizes (one per individual)
    step_sizes = np.full((mu, dimension), INITIAL_SIGMA)

    return positions, step_sizes


def intermediate_recombination(parents_x: np.ndarray, parents_sigma: np.ndarray) -> tuple:
    """
    Perform intermediate (global) recombination.

    Args:
        parents_x: Parent positions
        parents_sigma: Parent step sizes

    Returns:
        Recombined position and step size
    """
    # Randomly select two parents
    idx = np.random.choice(len(parents_x), 2, replace=False)

    # Average the two parents
    offspring_x = (parents_x[idx[0]] + parents_x[idx[1]]) / 2
    offspring_sigma = (parents_sigma[idx[0]] + parents_sigma[idx[1]]) / 2

    return offspring_x, offspring_sigma


def discrete_recombination(parents_x: np.ndarray, parents_sigma: np.ndarray) -> tuple:
    """
    Perform discrete (global) recombination.

    Args:
        parents_x: Parent positions
        parents_sigma: Parent step sizes

    Returns:
        Recombined position and step size
    """
    dimension = parents_x.shape[1]

    offspring_x = np.zeros(dimension)
    offspring_sigma = np.zeros(dimension)

    for i in range(dimension):
        # Randomly select a parent for each dimension
        parent_idx = np.random.randint(len(parents_x))
        offspring_x[i] = parents_x[parent_idx, i]
        offspring_sigma[i] = parents_sigma[parent_idx, i]

    return offspring_x, offspring_sigma


def mutate(x: np.ndarray, sigma: np.ndarray, dimension: int) -> tuple:
    """
    Perform self-adaptive mutation.

    Args:
        x: Individual position
        sigma: Step sizes
        dimension: Problem dimension

    Returns:
        Mutated position and step sizes
    """
    # Learning rates for self-adaptation
    tau = 1.0 / np.sqrt(2.0 * dimension)  # Global learning rate
    tau_prime = 1.0 / np.sqrt(2.0 * np.sqrt(dimension))  # Coordinate-wise learning rate

    # Mutate step sizes first (self-adaptation)
    global_factor = np.exp(tau_prime * np.random.randn())
    local_factors = np.exp(tau * np.random.randn(dimension))
    sigma_new = sigma * global_factor * local_factors

    # Ensure step sizes don't become too small
    sigma_new = np.maximum(sigma_new, 1e-10)

    # Mutate position using new step sizes
    x_new = x + sigma_new * np.random.randn(dimension)

    # Clip to bounds [-5, 5] for BBOB problems
    x_new = np.clip(x_new, -5.0, 5.0)

    return x_new, sigma_new


def studentnumber1_studentnumber2_ES(problem, mu: int = MU, lambda_: int = LAMBDA,
                                     use_plus_selection: bool = False):
    """
    Evolution Strategy with self-adaptive step sizes.

    Implements (μ, λ)-ES or (μ + λ)-ES for continuous optimization.

    Args:
        problem: IOH problem instance
        mu: Number of parents
        lambda_: Number of offspring
        use_plus_selection: If True, use (μ + λ) selection; otherwise use (μ, λ)
    """
    dimension = problem.meta_data.n_variables

    # Initialize population
    parents_x, parents_sigma = initialize_population(mu, dimension)

    # Evaluate initial population
    parents_fitness = np.array([problem(x) for x in parents_x])

    # Track best solution
    best_idx = np.argmin(parents_fitness)  # BBOB problems are minimization
    best_x = parents_x[best_idx].copy()
    best_fitness = parents_fitness[best_idx]

    # Main evolutionary loop
    while problem.state.evaluations < budget:
        # Generate offspring
        offspring_x = []
        offspring_sigma = []
        offspring_fitness = []

        for _ in range(lambda_):
            if problem.state.evaluations >= budget:
                break

            # Recombination
            child_x, child_sigma = intermediate_recombination(parents_x, parents_sigma)

            # Mutation
            child_x, child_sigma = mutate(child_x, child_sigma, dimension)

            # Evaluation
            fitness = problem(child_x)

            offspring_x.append(child_x)
            offspring_sigma.append(child_sigma)
            offspring_fitness.append(fitness)

            # Update best solution
            if fitness < best_fitness:
                best_fitness = fitness
                best_x = child_x.copy()

        if len(offspring_x) == 0:
            break

        offspring_x = np.array(offspring_x)
        offspring_sigma = np.array(offspring_sigma)
        offspring_fitness = np.array(offspring_fitness)

        # Selection
        if use_plus_selection:
            # (μ + λ) selection: combine parents and offspring
            combined_x = np.vstack([parents_x, offspring_x])
            combined_sigma = np.vstack([parents_sigma, offspring_sigma])
            combined_fitness = np.concatenate([parents_fitness, offspring_fitness])

            # Select best μ individuals
            sorted_indices = np.argsort(combined_fitness)[:mu]
            parents_x = combined_x[sorted_indices]
            parents_sigma = combined_sigma[sorted_indices]
            parents_fitness = combined_fitness[sorted_indices]
        else:
            # (μ, λ) selection: select from offspring only
            if len(offspring_fitness) >= mu:
                sorted_indices = np.argsort(offspring_fitness)[:mu]
                parents_x = offspring_x[sorted_indices]
                parents_sigma = offspring_sigma[sorted_indices]
                parents_fitness = offspring_fitness[sorted_indices]


def create_problem(fid: int):
    """
    Create an IOH problem instance with logger.

    Args:
        fid: Function ID (23 for Katsuura)

    Returns:
        Problem instance and logger
    """
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.BBOB)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root="data",  # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
        folder_name="run",  # the folder name to which the raw performance data will be stored
        algorithm_name="evolution_strategy",  # name of your algorithm
        algorithm_info="Practical assignment part2 of the EA course",
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l


if __name__ == "__main__":
    # this how you run your algorithm with 20 repetitions/independent run
    print("=" * 60)
    print("Running Evolution Strategy on F23 (Katsuura)")
    print("=" * 60)
    print(f"Problem: F23 (Katsuura) from BBOB")
    print(f"Dimension: {dimension}")
    print(f"Budget: {budget:,} function evaluations")
    print(f"Strategy: ({MU}, {LAMBDA})-ES with self-adaptive step sizes")
    print(f"Number of runs: 20")
    print("=" * 60)
    print()

    F23, _logger = create_problem(23)

    best_results = []

    for run in range(20):
        print(f"Run {run + 1}/20...", end=' ')
        studentnumber1_studentnumber2_ES(F23)
        best_fitness = F23.state.current_best.y
        best_results.append(best_fitness)
        print(f"Best fitness: {best_fitness:.6e}")
        F23.reset() # it is necessary to reset the problem after each independent run

    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder

    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Mean best fitness: {np.mean(best_results):.6e}")
    print(f"Std best fitness:  {np.std(best_results):.6e}")
    print(f"Min best fitness:  {np.min(best_results):.6e}")
    print(f"Max best fitness:  {np.max(best_results):.6e}")
    print("=" * 60)
    print("\nAll runs completed! Check the 'data/run' folder for IOH analyzer data.")
