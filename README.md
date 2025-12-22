# Evolutionary Algorithms - Practical Assignment

This repository contains the implementation of a Genetic Algorithm (GA) and Evolution Strategy (ES) for solving optimization problems from the IOHprofiler benchmark suite.

## Assignment Overview

### Part 1: Genetic Algorithm (GA)
- **Problems**:
  - F18: Low Autocorrelation Binary Sequences (LABS) - Dimension 50
  - F23: N-Queens Problem - Dimension 49
- **Budget**: 5,000 function evaluations per problem
- **Runs**: 20 independent runs per problem
- **Challenge**: Same hyperparameters must work well for both problems

### Part 2: Evolution Strategy (ES)
- **Problem**: F23 Katsuura (BBOB benchmark) - Dimension 10
- **Budget**: 50,000 function evaluations
- **Runs**: 20 independent runs
- **Method**: (μ, λ)-ES with self-adaptive step sizes

## Files Structure

```
.
├── GA.py                  # Genetic Algorithm implementation
├── tuning.py             # Hyperparameter tuning for GA
├── ES.py                 # Evolution Strategy implementation
├── README.md             # This file
└── data/                 # Output directory (created automatically)
    └── run/              # IOH analyzer data files
```

## Installation

### Requirements
- Python 3.7+
- NumPy 2.0+
- IOHexperimenter

### Install Dependencies

```bash
pip install ioh numpy
```

## Usage

### 1. Hyperparameter Tuning (Part 1)

First, run the hyperparameter tuning to find optimal parameters for the GA:

```bash
python3 tuning.py
```

This will:
- Test multiple hyperparameter configurations
- Use 100,000 function evaluations budget
- Find parameters that work well on both F18 and F23
- Print the best configuration

**Expected output:**
- Best population size
- Best mutation rate
- Best crossover rate
- Best tournament size

### 2. Running the Genetic Algorithm (Part 1)

After tuning, update the hyperparameters in `GA.py` (lines 16-19) with the best values, then run:

```bash
python3 GA.py
```

This will:
- Run GA on F18 (LABS) for 20 independent runs
- Run GA on F23 (N-Queens) for 20 independent runs
- Generate IOH analyzer data in `data/run/`

### 3. Running the Evolution Strategy (Part 2)

```bash
python3 ES.py
```

This will:
- Run ES on F23 (Katsuura) for 20 independent runs
- Generate IOH analyzer data in `data/run/`
- Print statistics (mean, std, min, max) of best fitness values

## Algorithm Details

### Genetic Algorithm (GA)

**Representation**: Binary strings

**Operators**:
- **Selection**: Tournament selection (configurable tournament size)
- **Crossover**: Uniform crossover
- **Mutation**: Bit-flip mutation with configurable rate
- **Elitism**: Best individual always survives

**Default Hyperparameters** (before tuning):
- Population size: 100
- Mutation rate: 0.05
- Crossover rate: 0.9
- Tournament size: 3

### Evolution Strategy (ES)

**Representation**: Real-valued vectors

**Strategy**: (10, 70)-ES (10 parents, 70 offspring)

**Operators**:
- **Recombination**: Intermediate (global) recombination
- **Mutation**: Gaussian mutation with self-adaptive step sizes
- **Selection**: Comma strategy (select from offspring only)

**Self-Adaptation**:
- Learning rate τ = 1/√(2n)
- Learning rate τ' = 1/√(2√n)
- Individual step sizes per dimension

## Hyperparameter Tuning Strategy

The tuning procedure uses a strategic grid search:

1. **Search Space**:
   - Population sizes: [50, 100, 150]
   - Mutation rates: [0.01, 0.03, 0.05, 0.08]
   - Crossover rates: [0.7, 0.85, 0.95]
   - Tournament sizes: [2, 3, 5]

2. **Budget Allocation**:
   - 2 runs per configuration per problem
   - Evaluates on both F18 and F23
   - Selects configuration with best combined score

3. **Objective**:
   - Maximize average performance on both problems
   - Ensures robustness across different problem types

## Output Files

All algorithms generate IOH analyzer compatible data files in `data/run/`:

- `IOHprofiler_f18_DIM50.dat` - LABS problem results
- `IOHprofiler_f23_DIM49.dat` - N-Queens problem results
- `IOHprofiler_f23_DIM10.dat` - Katsuura problem results

These can be uploaded to [IOHanalyzer](https://iohanalyzer.liacs.nl) for visualization and analysis.

## Results Analysis

After running the algorithms, you can:

1. **Upload to IOHanalyzer**:
   - Compress the `data/run` folder
   - Upload to https://iohanalyzer.liacs.nl
   - Visualize ERT, ECDF curves, and convergence plots

2. **View Console Output**:
   - Best fitness values per run
   - Summary statistics (mean, std, min, max)

## Implementation Highlights

### Key Features

1. **Fixed Random Seed**: Ensures reproducibility (seed=42)
2. **Budget Tracking**: Uses IOH's built-in evaluation counter
3. **Elitism**: GA preserves best solution across generations
4. **Self-Adaptation**: ES adapts step sizes during evolution
5. **Bounds Handling**: ES clips solutions to [-5, 5] for BBOB

### Design Decisions

**GA**:
- Uniform crossover chosen for better exploration on binary problems
- Tournament selection for selection pressure control
- Bit-flip mutation for local search

**ES**:
- (μ, λ) strategy chosen over (μ + λ) to avoid premature convergence
- Intermediate recombination for smooth search
- Self-adaptive step sizes for parameter-free optimization

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'ioh'**
   ```bash
   pip install ioh
   ```

2. **Memory issues during tuning**
   - Reduce the hyperparameter search space in `tuning.py`
   - Reduce number of runs per configuration

3. **Different results than report**
   - Ensure fixed random seed is set (np.random.seed(42))
   - Verify hyperparameter values match

## References

1. Doerr, C., et al. (2019). Benchmarking discrete optimization heuristics with IOHprofiler. GECCO.
2. de Nobel, J., et al. (2024). Iohexperimenter: Benchmarking platform for iterative optimization heuristics. Evolutionary Computation.
3. Finck, S., et al. (2010). Real-parameter black-box optimization benchmarking 2009: Presentation of the noiseless functions.

## Contact

For questions about the implementation, please refer to the practical assignment documentation or contact the course instructors.

---

**Last Updated**: December 22, 2025
**Course**: Evolutionary Algorithms, LIACS 2025-2026
