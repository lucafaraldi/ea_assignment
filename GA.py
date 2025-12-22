from typing import Tuple
import numpy as np
# you need to install this package `ioh`. Please see documentations here:
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
import ioh
from ioh import get_problem, logger, ProblemClass

budget = 5000

# Set random seed for reproducibility
np.random.seed(42)

# Global variables for hyperparameters (will be set by tuning)
POPULATION_SIZE = 100
MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.85
TOURNAMENT_SIZE = 5


def tournament_selection(population: np.ndarray, fitness: np.ndarray, tournament_size: int = 3) -> np.ndarray:
    """Select an individual using tournament selection."""
    indices = np.random.choice(len(population), tournament_size, replace=False)
    tournament_fitness = fitness[indices]
    winner_idx = indices[np.argmax(tournament_fitness)]
    return population[winner_idx].copy()


def uniform_crossover(parent1: np.ndarray, parent2: np.ndarray, crossover_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """Perform uniform crossover between two parents."""
    if np.random.random() > crossover_rate:
        return parent1.copy(), parent2.copy()

    mask = np.random.randint(0, 2, len(parent1), dtype=bool)
    offspring1 = np.where(mask, parent1, parent2)
    offspring2 = np.where(mask, parent2, parent1)
    return offspring1, offspring2


def bit_flip_mutation(individual: np.ndarray, mutation_rate: float) -> np.ndarray:
    """Perform bit-flip mutation on an individual."""
    mask = np.random.random(len(individual)) < mutation_rate
    individual[mask] = 1 - individual[mask]
    return individual


def initialize_population(pop_size: int, dimension: int) -> np.ndarray:
    """Initialize a random binary population."""
    return np.random.randint(0, 2, (pop_size, dimension))


def evaluate_population(population: np.ndarray, problem: ioh.problem.PBO) -> np.ndarray:
    """Evaluate all individuals in the population."""
    fitness = np.zeros(len(population))
    for i, individual in enumerate(population):
        fitness[i] = problem(individual)
    return fitness


def studentnumber1_studentnumber2_GA(problem: ioh.problem.PBO,
                                      population_size: int = POPULATION_SIZE,
                                      mutation_rate: float = MUTATION_RATE,
                                      crossover_rate: float = CROSSOVER_RATE,
                                      tournament_size: int = TOURNAMENT_SIZE) -> None:
    """
    Genetic Algorithm for binary optimization problems.

    Args:
        problem: IOH problem instance
        population_size: Size of the population
        mutation_rate: Probability of bit flip per bit
        crossover_rate: Probability of crossover
        tournament_size: Number of individuals in tournament selection
    """
    dimension = problem.meta_data.n_variables

    # Initialize population
    population = initialize_population(population_size, dimension)
    fitness = evaluate_population(population, problem)

    # Track best solution
    best_idx = np.argmax(fitness)
    best_solution = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    # Main evolutionary loop
    while problem.state.evaluations < budget:
        # Create offspring population
        offspring = []

        # Elitism: keep the best individual
        offspring.append(best_solution.copy())

        # Generate offspring
        while len(offspring) < population_size:
            # Selection
            parent1 = tournament_selection(population, fitness, tournament_size)
            parent2 = tournament_selection(population, fitness, tournament_size)

            # Crossover
            child1, child2 = uniform_crossover(parent1, parent2, crossover_rate)

            # Mutation
            child1 = bit_flip_mutation(child1, mutation_rate)
            child2 = bit_flip_mutation(child2, mutation_rate)

            offspring.append(child1)
            if len(offspring) < population_size:
                offspring.append(child2)

        # Update population
        population = np.array(offspring[:population_size])

        # Evaluate new population
        if problem.state.evaluations < budget:
            fitness = evaluate_population(population, problem)

            # Update best solution
            current_best_idx = np.argmax(fitness)
            if fitness[current_best_idx] > best_fitness:
                best_fitness = fitness[current_best_idx]
                best_solution = population[current_best_idx].copy()


def create_problem(dimension: int, fid: int) -> Tuple[ioh.problem.PBO, ioh.logger.Analyzer]:
    """
    Create an IOH problem instance with logger.

    Args:
        dimension: Problem dimension
        fid: Function ID (18 for LABS, 23 for N-Queens)

    Returns:
        Problem instance and logger
    """
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root="data",  # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
        folder_name="run",  # the folder name to which the raw performance data will be stored
        algorithm_name="genetic_algorithm",  # name of your algorithm
        algorithm_info="Practical assignment of the EA course",
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l


if __name__ == "__main__":
    # this how you run your algorithm with 20 repetitions/independent run
    print("Running GA on F18 (LABS) with dimension 50...")
    # create the LABS problem and the data logger
    F18, _logger = create_problem(dimension=50, fid=18)
    for run in range(20):
        print(f"  Run {run + 1}/20", end='\r')
        studentnumber1_studentnumber2_GA(F18)
        F18.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder
    print("\nF18 (LABS) completed!")

    print("\nRunning GA on F23 (N-Queens) with dimension 49...")
    # create the N-Queens problem and the data logger
    F23, _logger = create_problem(dimension=49, fid=23)
    for run in range(20):
        print(f"  Run {run + 1}/20", end='\r')
        studentnumber1_studentnumber2_GA(F23)
        F23.reset()
    _logger.close()
    print("\nF23 (N-Queens) completed!")
    print("\nAll runs completed! Check the 'data/run' folder for results.")
