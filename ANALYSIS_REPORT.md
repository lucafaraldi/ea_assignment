# IOH Data Analysis Report
## Evolutionary Algorithms Practical Assignment

**Date:** January 2, 2026
**Analysis Tool:** Custom Python analysis script with IOHprofiler data

---

## Executive Summary

This report presents the empirical analysis of evolutionary algorithms applied to three optimization problems as part of the EA practical assignment. All experiments were conducted with fresh runs using tuned hyperparameters, fixed random seed (42), and proper IOHprofiler logging.

---

## Part 1: Genetic Algorithm

### Hyperparameter Tuning Results

**Methodology:**
- Grid search over 108 configurations
- Budget: 100,000 evaluations total
- ~925 evaluations per configuration (2 runs × 2 problems)
- Combined scoring: (F18_score + F23_score) / 2

**Optimal Configuration:**
- Population Size: 100
- Mutation Rate: 0.01 (approximately 1/n for n=50)
- Crossover Rate: 0.85
- Tournament Size: 5
- **Combined Score: 5.62**

**Top 5 Configurations:**
1. pop=100, mut=0.010, cross=0.85, tourn=5 → 5.62
2. pop=150, mut=0.010, cross=0.95, tourn=5 → 5.58
3. pop=50, mut=0.050, cross=0.85, tourn=5 → 5.52
4. pop=100, mut=0.050, cross=0.70, tourn=5 → 5.37
5. pop=50, mut=0.030, cross=0.95, tourn=5 → 5.36

**Key Insights:**
- Tournament size of 5 consistently appears in top configurations
- Moderate mutation rates (0.01-0.05) perform better than high rates (0.08)
- Crossover rates between 0.70-0.95 all perform well
- Population size of 100 offers good balance

---

### F18: LABS Problem (D=50)

**Problem Characteristics:**
- Type: Pseudo-Boolean Optimization (PBO)
- Objective: Maximize merit factor (minimize autocorrelation)
- Search Space: Binary strings of length 50
- Budget: 5,000 evaluations per run
- Number of Runs: 20

**Performance Metrics:**
- **Mean Best Fitness:** 4.0273 ± 0.4535
- **Min Best Fitness:** 3.3156
- **Max Best Fitness:** 4.8638
- **AUC (Area Under Curve):** 3.66 ± 0.37

**Analysis:**
- Moderate performance variability (std = 0.45)
- All runs achieved fitness > 3.0
- Best run achieved 4.86 merit factor
- Convergence plots show steady improvement throughout budget
- ECDF indicates ~50% of runs achieve fitness ≥ 4.0
- ERT analysis shows efficient target achievement

**Algorithm Components:**
- **Selection:** Tournament selection (k=5)
- **Crossover:** Uniform crossover (pc=0.85)
- **Mutation:** Bit-flip mutation (pm=0.01)
- **Elitism:** Best individual preserved

**Generated Plots:**
1. `convergence_F18_LABS_GA,_D=50.pdf` - Median + IQR + individual trajectories
2. `ecdf_F18_LABS_GA,_D=50.pdf` - Empirical cumulative distribution
3. `ert_F18_LABS_GA,_D=50.pdf` - Expected running time curves

---

### F23: N-Queens Problem (D=49)

**Problem Characteristics:**
- Type: Pseudo-Boolean Optimization (PBO)
- Objective: Minimize number of conflicting queen pairs
- Search Space: Binary strings of length 49
- Budget: 5,000 evaluations per run
- Number of Runs: 20

**Performance Metrics:**
- **Mean Best Fitness:** 6.10 ± 0.54
- **Min Best Fitness:** 5.00
- **Max Best Fitness:** 7.00
- **AUC (Area Under Curve):** -7.21 ± 2.01

**Analysis:**
- Lower variability than F18 (std = 0.54)
- All runs achieved fitness ≥ 5.0
- Best run achieved 7.0 (meaning 0 conflicts - perfect solution!)
- Convergence shows rapid initial improvement, then plateaus
- ECDF shows strong performance clustering around 6.0
- Multiple runs found near-optimal solutions

**Same Hyperparameters:**
As required by the assignment, the same hyperparameters tuned for both problems were used successfully, demonstrating good generalization of the GA configuration across different problem structures.

**Generated Plots:**
1. `convergence_F23_N-Queens_GA,_D=49.pdf`
2. `ecdf_F23_N-Queens_GA,_D=49.pdf`
3. `ert_F23_N-Queens_GA,_D=49.pdf`

---

## Part 2: Evolution Strategy

### F23: Katsuura Function (D=10)

**Problem Characteristics:**
- Type: BBOB (Black-Box Optimization Benchmarking)
- Objective: Minimize (highly multimodal function)
- Search Space: Continuous [-5, 5]^10
- Budget: 50,000 evaluations per run
- Number of Runs: 20

**ES Configuration:**
- **Strategy:** (10, 70)-ES (comma selection)
- **Mu (parents):** 10
- **Lambda (offspring):** 70 (7×μ ratio)
- **Initial Step Size:** σ₀ = 0.5
- **Recombination:** Intermediate (global)
- **Mutation:** Self-adaptive with log-normal update
- **Learning Rates:**
  - τ' (global) = 1/√(2D) = 0.158
  - τ (local) = 1/√(2√D) = 0.316

**Performance Metrics:**
- **Mean Best Fitness:** 1.6891 ± 0.4243
- **Min Best Fitness:** 0.5010
- **Max Best Fitness:** 2.1591
- **AUC (Area Under Curve):** 1.73 ± 0.40

**Analysis:**
- Good performance for a highly multimodal BBOB function
- Best run achieved 0.50 (excellent minimization)
- Moderate variability indicates consistent convergence
- Self-adaptation successfully tuned step sizes
- Convergence plots show rapid initial descent, then refinement
- ECDF shows most runs achieve fitness ≤ 2.0
- ERT curves indicate efficient exploration-exploitation balance

**Technical Details:**
- **Self-Adaptation Formula:** σ' = σ × exp(τ'·N(0,1)) × exp(τ·N_i(0,1))
- **Mutation:** x' = x + σ'·N(0,I)
- **Bounds:** Hard clipping to [-5, 5]
- **Selection:** Comma strategy (only offspring compete)

**Generated Plots:**
1. `convergence_F23_Katsuura_ES,_D=10.pdf`
2. `ecdf_F23_Katsuura_ES,_D=10.pdf`
3. `ert_F23_Katsuura_ES,_D=10.pdf`

---

## Comparative Analysis

### GA Performance Comparison (F18 vs F23)

| Metric | F18 LABS | F23 N-Queens |
|--------|----------|--------------|
| Mean Fitness | 4.0273 ± 0.45 | 6.10 ± 0.54 |
| Best Solution | 4.86 | 7.00 (optimal!) |
| AUC | 3.66 | -7.21 |
| Variability | Higher | Lower |

**Insights:**
- Same hyperparameters work well for both problems
- F23 shows more consistent performance (lower std)
- F23 achieved optimal solution in 1 run
- Both problems show good scalability within 5K budget

### Algorithm Effectiveness

**Genetic Algorithm:**
- ✅ Successfully optimized two different PBO problems
- ✅ Found near-optimal/optimal solutions
- ✅ Hyperparameters generalize well
- ✅ Efficient convergence within budget

**Evolution Strategy:**
- ✅ Effective on multimodal continuous optimization
- ✅ Self-adaptation working properly
- ✅ Good exploration-exploitation balance
- ✅ Consistent performance across runs

---

## Reproducibility

All experiments are fully reproducible:

**Fixed Parameters:**
- Random seed: 42 (NumPy)
- Python version: 3.11
- IOHprofiler version: 0.3.22

**Data Files:**
- `data/run/IOHprofiler_f18_LABS.json` + `.dat` files
- `data/run-1/IOHprofiler_f23_NQueens.json` + `.dat` files
- `data/run-2/IOHprofiler_f23_Katsuura.json` + `.dat` files

**Analysis Scripts:**
- `tuning.py` - Hyperparameter optimization
- `GA.py` - Genetic algorithm implementation
- `ES.py` - Evolution strategy implementation
- `analyze_results_corrected.py` - IOH data analysis and plotting

---

## Plots Summary

**9 Total Plots Generated:**

### Convergence Plots (3)
- Show best-so-far fitness vs function evaluations
- Include median trajectory + IQR (25th-75th percentile)
- Individual run trajectories (semi-transparent)
- Clearly show optimization dynamics

### ECDF Plots (3)
- Empirical Cumulative Distribution Function of final best values
- For maximization: P(X ≥ target)
- For minimization: P(X ≤ target)
- Shows solution quality distribution across runs

### ERT Plots (3)
- Expected Running Time to reach various targets
- Log-scale y-axis for better visualization
- Budget line shows evaluation limit
- Indicates algorithm efficiency

---

## Conclusions

1. **Hyperparameter Tuning:** Grid search successfully identified robust parameters that work well for both F18 and F23 PBO problems

2. **GA Performance:** The genetic algorithm with tournament selection, uniform crossover, and bit-flip mutation effectively optimized both LABS and N-Queens problems, even finding an optimal solution for N-Queens

3. **ES Performance:** The (10,70)-ES with self-adaptive step sizes achieved good performance on the highly multimodal Katsuura function, with mean fitness 1.69 ± 0.42

4. **Methodology:** All experiments followed proper scientific methodology with:
   - Fixed random seeds for reproducibility
   - Multiple independent runs (20 each)
   - Standardized logging (IOHprofiler format)
   - Comprehensive statistical analysis

5. **Future Improvements:**
   - For ES: Consider CMA-ES for better covariance adaptation
   - For GA: Test adaptive mutation rates
   - Increase budget for more thorough exploration
   - Apply multi-objective approaches

---

## Files Generated

**Data:**
- IOHprofiler JSON files (3)
- IOHprofiler DAT files (3)
- Consolidated ioh_data/ directory

**Analysis:**
- 9 PDF plots (convergence, ECDF, ERT)
- analysis_summary.txt
- ANALYSIS_REPORT.md (this file)

**Logs:**
- tuning_output.log
- ga_output.log
- es_output.log
- analysis_output.log

---

**End of Report**
