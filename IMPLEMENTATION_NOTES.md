# Implementation Notes - EA Assignment

## Overview

This document provides detailed technical notes about the implementation choices, design decisions, and algorithmic details for both the Genetic Algorithm (Part 1) and Evolution Strategy (Part 2).

## Part 1: Genetic Algorithm

### Problem Analysis

#### F18: Low Autocorrelation Binary Sequences (LABS)
- **Type**: Non-linear binary optimization
- **Dimension**: 50
- **Objective**: Maximize reciprocal of autocorrelation (higher is better)
- **Characteristics**:
  - Highly multimodal landscape
  - Strong epistatic interactions between bits
  - Theoretical maximum merit factor ≈ 12.32 for large n

#### F23: N-Queens Problem
- **Type**: Constraint satisfaction problem (as optimization)
- **Dimension**: 49 (49 queens on 49×49 board)
- **Objective**: Maximize number of non-attacking queen pairs
- **Characteristics**:
  - Discrete, combinatorial problem
  - Many local optima
  - Multiple global optima (different valid solutions)

### Algorithm Design

#### 1. Representation
- **Encoding**: Direct binary representation
- **F18**: Each bit represents 0 or 1 in the sequence
- **F23**: Binary encoding of queen positions (may need decoding)

#### 2. Selection: Tournament Selection
```python
def tournament_selection(population, fitness, tournament_size=3):
    # Select k random individuals
    # Return the best among them
```

**Why Tournament Selection?**
- Simple and efficient O(k) complexity
- Tunable selection pressure via tournament size
- Works well without fitness scaling
- Maintains diversity better than roulette wheel
- No issues with negative fitness values

**Tournament Size Effects**:
- Small (2): Low selection pressure, more exploration
- Medium (3-5): Balanced exploration-exploitation
- Large (>5): High selection pressure, faster convergence

#### 3. Crossover: Uniform Crossover
```python
def uniform_crossover(parent1, parent2, crossover_rate):
    # For each bit position:
    #   - With 50% probability, swap bits between parents
```

**Why Uniform Crossover?**
- Better for problems with tight linkage (LABS)
- Provides more disruptive recombination than one-point
- Explores wider offspring space
- Position-independent mixing
- Empirically performs well on binary problems

**Alternatives Considered**:
- One-point: Too conservative, preserves blocks too much
- Two-point: Better than one-point but still position-dependent
- **Chosen: Uniform** - Best for unknown epistatic structure

#### 4. Mutation: Bit-Flip Mutation
```python
def bit_flip_mutation(individual, mutation_rate):
    # For each bit:
    #   - With probability mutation_rate, flip the bit
```

**Why Bit-Flip?**
- Standard mutation for binary strings
- Unbiased (no preference for 0 or 1)
- Controlled by mutation rate parameter

**Mutation Rate Guidelines**:
- Typical: 1/n (n = problem dimension)
- For n=50: 1/50 = 0.02
- Range tested: 0.01 to 0.08
- **Default: 0.05** (slightly higher for more exploration)

#### 5. Population Management

**Elitism**:
```python
offspring.append(best_solution.copy())  # Always preserve best
```
- **Why**: Guarantees monotonic improvement
- Prevents loss of good solutions due to randomness
- Minimal computational overhead

**Generational Replacement**:
- Full generational replacement (all offspring replace all parents)
- Offspring population size = parent population size
- No steady-state replacement

#### 6. Population Size

**Tested Range**: [50, 100, 150]

**Trade-offs**:
- **Small (50)**:
  - Faster iterations
  - More generations within budget
  - Risk of premature convergence

- **Medium (100)**:
  - Good balance
  - Sufficient diversity
  - **Default choice**

- **Large (150)**:
  - Better diversity
  - Fewer generations within budget
  - Higher computational cost per generation

**Formula Guidance**:
- Typical GA: 50-500 individuals
- Rule of thumb: √(search space size) ≈ √(2^50) ≈ 2^25 (way too large)
- Practical: 2-10 times problem dimension
- For n=50: 100-500 → **chosen 100**

### Hyperparameter Tuning Strategy

#### Tuning Objective
```python
combined_score = (f18_score + f23_score) / 2
```

**Why Average?**
- Ensures good performance on BOTH problems
- Prevents over-specialization to one problem
- Simple and interpretable

**Alternative Objectives Considered**:
- Minimum: Too conservative, optimizes worst case
- Weighted sum: Arbitrary weight choice
- **Chosen: Average** - Fair and balanced

#### Budget Allocation

Total budget: 100,000 function evaluations

**Strategy**:
```
- Number of configurations: 3 × 4 × 3 × 3 = 108
- Runs per config: 2 runs × 2 problems = 4 runs
- Evals per run: 5,000
- Total per config: 20,000
- Maximum configs testable: 100,000 / 20,000 = 5
```

**Decision**: Test subset of configurations
- Sample most promising regions of hyperparameter space
- Use 2 runs per problem (instead of 20) for tuning
- Reserve full 20 runs for final evaluation

#### Search Space Design

```python
hyperparameter_space = {
    "population_size": [50, 100, 150],      # 3 values
    "mutation_rate": [0.01, 0.03, 0.05, 0.08],  # 4 values
    "crossover_rate": [0.7, 0.85, 0.95],    # 3 values
    "tournament_size": [2, 3, 5],           # 3 values
}
```

**Ranges Based On**:
- Literature recommendations
- EA theory (mutation rate ≈ 1/n)
- Empirical experience with binary GAs
- Avoiding extreme values

### Expected Performance

**F18 (LABS, n=50)**:
- Known best value: 8.1699
- Most optimizers: ~7.0
- GA target: > 2.5 (merit factor)

**F23 (N-Queens, n=49)**:
- Optimal: 0 conflicts (many solutions exist)
- Near-optimal: < 5 conflicts
- GA should find perfect solution

## Part 2: Evolution Strategy

### Problem Analysis

#### F23: Katsuura Function (BBOB)
- **Type**: Continuous optimization, multimodal
- **Dimension**: 10
- **Domain**: [-5, 5]^10
- **Objective**: Minimize (BBOB problems are minimization)
- **Characteristics**:
  - Highly irregular landscape
  - Many local optima
  - Non-separable
  - Rotation and scaling applied

### Algorithm Design: (μ, λ)-ES

#### Strategy Notation
- **(μ, λ)-ES**: Select μ parents from λ offspring only
- μ = 10 (number of parents)
- λ = 70 (number of offspring)
- Ratio: λ/μ = 7 (typical range: 5-7)

**Why (μ, λ) instead of (μ + λ)?**
- **Comma strategy**: Parents not reconsidered
  - Forces progress through offspring
  - Prevents premature convergence
  - Better for time-varying fitness
  - Self-adaptation works better

- **Plus strategy**: Parents compete with offspring
  - More exploitative
  - Risk of stagnation
  - Less suitable for self-adaptation

#### 1. Representation
```python
individual = {
    'x': [x1, x2, ..., x10],          # Position in R^10
    'sigma': [σ1, σ2, ..., σ10]       # Step sizes per dimension
}
```

**Why individual step sizes?**
- Different dimensions may need different exploration ranges
- Adaptation to local landscape curvature
- More parameters, but better convergence

#### 2. Initialization
```python
positions = uniform(-5, 5, size=(mu, dim))
step_sizes = 0.5 * ones((mu, dim))
```

**Initial Step Size Choice**:
- σ₀ = 0.5
- Domain width: 10 (from -5 to 5)
- σ₀/domain ≈ 5% (reasonable starting exploration)
- Will adapt during evolution

#### 3. Recombination: Intermediate (Global)
```python
def intermediate_recombination(parents_x, parents_sigma):
    # Select 2 random parents
    # offspring = (parent1 + parent2) / 2
```

**Why Intermediate?**
- Creates offspring in convex hull of parents
- Smooth exploration of search space
- Works well with continuous domains
- Center bias helps convergence

**Alternative: Discrete Recombination**
- Also implemented as option
- Each gene from random parent
- More disruptive
- Can help escape local optima

#### 4. Mutation: Self-Adaptive Gaussian

```python
τ' = 1 / sqrt(2 * n)           # Global learning rate
τ = 1 / sqrt(2 * sqrt(n))       # Coordinate-wise learning rate

# Mutate step sizes (log-normal update)
σ'ᵢ = σᵢ * exp(τ' * N(0,1) + τ * Nᵢ(0,1))

# Mutate position
x'ᵢ = xᵢ + σ'ᵢ * Nᵢ(0,1)
```

**Why Self-Adaptation?**
- No manual step size tuning needed
- Automatically adjusts to landscape
- **1/5 success rule** emergent behavior
- Step sizes evolve with population

**Learning Rate Formulas** (Schwefel, 1995):
- τ': Overall magnitude of step size changes
- τ: Allows different rates per dimension
- Standard formulas for n-dimensional problems

**Step Size Bounds**:
```python
sigma_new = max(sigma_new, 1e-10)  # Prevent collapse to zero
```

#### 5. Selection
```python
# Evaluate all λ offspring
# Sort by fitness
# Select best μ as new parents
```

**Truncation Selection**:
- Deterministic
- High selection pressure
- Elitist among offspring
- Compatible with comma strategy

#### 6. Bounds Handling
```python
x_new = np.clip(x_new, -5.0, 5.0)
```

**Simple Clipping**:
- Keep solutions in valid domain
- Alternative: Reflection or toroidal wrap
- Clipping is simplest and works well for BBOB

### Parameter Settings

| Parameter | Value | Justification |
|-----------|-------|---------------|
| μ (parents) | 10 | ~√dim rule (√10 ≈ 3, use 10 for robustness) |
| λ (offspring) | 70 | 7×μ (standard ratio) |
| σ₀ (initial step size) | 0.5 | 5% of domain width |
| τ' | 1/√(2n) ≈ 0.158 | Standard self-adaptation rate |
| τ | 1/√(2√n) ≈ 0.281 | Standard self-adaptation rate |

### Expected Performance

**F23 Katsuura (BBOB, dim=10)**:
- Function is shifted/rotated
- Optimal value: f(x*) ≈ f_opt (varies by instance)
- Target: Achieve low function value
- ES with 50,000 evals should find good solutions

## Implementation Quality Assurance

### Testing Performed

1. **Unit Tests** (Informal):
   - Selection: Tournament returns valid individual
   - Crossover: Offspring are valid binary strings
   - Mutation: Maintains binary values
   - ES mutation: Step sizes stay positive

2. **Integration Tests**:
   - GA runs without errors on F18 and F23
   - ES runs without errors on Katsuura
   - Budget limits respected
   - Logger files created correctly

3. **Smoke Tests**:
   ```bash
   python3 -c "from GA import *; test_ga()"
   python3 -c "from ES import *; test_es()"
   ```

### Code Quality

- **Docstrings**: All major functions documented
- **Type Hints**: Function signatures typed
- **Comments**: Complex sections explained
- **Naming**: Descriptive variable names
- **Structure**: Modular, reusable functions

### Reproducibility

```python
np.random.seed(42)  # Fixed seed in all files
```

- Same seed across all scripts
- Deterministic results
- Report results match code output

## Common Pitfalls Avoided

### Genetic Algorithm
1. ✅ Maintained diversity (tournament selection, not greedy)
2. ✅ Preserved best solution (elitism)
3. ✅ Correct crossover rate usage (probability per pair)
4. ✅ Independent bit mutations (not swap mutation)
5. ✅ Budget tracking (via IOH counter)

### Evolution Strategy
1. ✅ Mutated step sizes before positions
2. ✅ Used log-normal update (multiplicative, not additive)
3. ✅ Prevented step size collapse (minimum bound)
4. ✅ Separated global and local learning rates
5. ✅ Respected bounds (clipping)

## Performance Optimization

### Algorithmic
- NumPy vectorization where possible
- Avoid repeated function evaluations
- Early termination if optimal found (optional)

### Implementation
- Pre-allocate arrays
- Avoid deep copying unless necessary
- Efficient selection (no sorting if not needed)

## Extensions and Future Work

### Potential Improvements

**GA**:
- Adaptive mutation rate (decrease over time)
- Island models (parallel populations)
- Specialized operators for N-Queens (permutation encoding)
- Linkage learning (identify building blocks)

**ES**:
- CMA-ES (Covariance Matrix Adaptation)
- Derandomization techniques
- Restart strategies
- Multi-parent recombination

**Tuning**:
- Bayesian optimization for hyperparameter search
- Racing algorithms (early stopping for bad configs)
- Meta-learning across problem instances

## References

### Books
1. Eiben, A. E., & Smith, J. E. (2015). Introduction to Evolutionary Computing. Springer.
2. Schwefel, H. P. (1995). Evolution and Optimum Seeking. Wiley.
3. Beyer, H. G., & Schwefel, H. P. (2002). Evolution strategies. Natural Computing, 1(1), 3-52.

### Papers
1. Doerr, C., et al. (2019). Benchmarking discrete optimization heuristics with IOHprofiler. GECCO.
2. Hansen, N., & Ostermeier, A. (2001). Completely derandomized self-adaptation in evolution strategies. Evolutionary Computation, 9(2), 159-195.
3. Spears, W. M., & De Jong, K. A. (1991). On the virtues of parameterized uniform crossover. ICGA.

### Online Resources
- IOHprofiler: https://iohprofiler.github.io/
- IOHanalyzer: https://iohanalyzer.liacs.nl/
- BBOB Workshop: https://coco.gforge.inria.fr/

---

**Document Version**: 1.0
**Last Updated**: December 22, 2025
**Authors**: Implementation for EA Course 2025-2026
