#!/usr/bin/env python3
"""
Convenience script to run all experiments for the Evolutionary Algorithms assignment.

This script orchestrates the complete workflow:
1. (Optional) Run hyperparameter tuning
2. Run GA on F18 and F23
3. Run ES on F23 Katsuura
4. Generate summary statistics
"""

import os
import sys
import argparse
import subprocess


def run_command(cmd, description):
    """Run a command and print status."""
    print("\n" + "=" * 70)
    print(f"RUNNING: {description}")
    print("=" * 70)
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"ERROR: {description} failed with code {result.returncode}")
        sys.exit(1)
    print(f"SUCCESS: {description} completed")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description='Run Evolutionary Algorithms experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run only GA and ES (skip tuning):
  python3 run_experiments.py --skip-tuning

  # Run complete workflow including tuning:
  python3 run_experiments.py --with-tuning

  # Run only tuning:
  python3 run_experiments.py --tuning-only

  # Run only GA:
  python3 run_experiments.py --ga-only

  # Run only ES:
  python3 run_experiments.py --es-only
        '''
    )

    parser.add_argument('--with-tuning', action='store_true',
                       help='Run hyperparameter tuning first (default: skip)')
    parser.add_argument('--skip-tuning', action='store_true',
                       help='Skip hyperparameter tuning (default)')
    parser.add_argument('--tuning-only', action='store_true',
                       help='Only run tuning, skip GA and ES')
    parser.add_argument('--ga-only', action='store_true',
                       help='Only run GA')
    parser.add_argument('--es-only', action='store_true',
                       help='Only run ES')

    args = parser.parse_args()

    # Default behavior: skip tuning unless --with-tuning specified
    run_tuning = args.with_tuning or args.tuning_only
    run_ga = not args.es_only and not args.tuning_only
    run_es = not args.ga_only and not args.tuning_only

    print("=" * 70)
    print("EVOLUTIONARY ALGORITHMS - EXPERIMENT RUNNER")
    print("=" * 70)
    print(f"Run Tuning: {run_tuning}")
    print(f"Run GA: {run_ga}")
    print(f"Run ES: {run_es}")
    print("=" * 70)

    # Step 1: Hyperparameter Tuning (Optional)
    if run_tuning:
        print("\n⚠️  WARNING: Hyperparameter tuning will take significant time!")
        print("   Budget: 100,000 function evaluations")
        print("   Estimated time: 10-30 minutes depending on hardware\n")

        response = input("Continue with tuning? [y/N]: ")
        if response.lower() != 'y':
            print("Skipping tuning. Using default parameters.")
        else:
            run_command("python3 tuning.py", "Hyperparameter Tuning")
            print("\n" + "=" * 70)
            print("TUNING COMPLETE!")
            print("=" * 70)
            print("Please update GA.py with the best hyperparameters shown above.")
            print("Edit lines 16-19 in GA.py with the tuned values.")
            print("=" * 70)

            if args.tuning_only:
                print("\nTuning-only mode: Exiting now.")
                return

            response = input("\nHave you updated GA.py? Continue? [y/N]: ")
            if response.lower() != 'y':
                print("Please update GA.py and run again.")
                return

    # Step 2: Run Genetic Algorithm
    if run_ga:
        run_command("python3 GA.py", "Part 1 - Genetic Algorithm (F18 & F23)")

    # Step 3: Run Evolution Strategy
    if run_es:
        run_command("python3 ES.py", "Part 2 - Evolution Strategy (F23 Katsuura)")

    # Summary
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nResults are saved in: data/run/")
    print("\nNext steps:")
    print("1. Compress the 'data/run' folder")
    print("2. Upload to IOHanalyzer: https://iohanalyzer.liacs.nl")
    print("3. Generate plots for your report (ERT, ECDF curves)")
    print("=" * 70)


if __name__ == "__main__":
    main()
