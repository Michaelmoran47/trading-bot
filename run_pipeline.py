#!/usr/bin/env python3
"""
Master script to run the entire ML trading pipeline
"""

import subprocess
import sys
import os

# Color codes for output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    """Print a colored header"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}{Colors.END}\n")

def run_script(script_name, description):
    """
    Run a Python script and handle errors
    
    Args:
        script_name: Name of the script to run
        description: What this script does
    """
    print_header(f"STEP: {description}")
    print(f"{Colors.YELLOW}Running: {script_name}{Colors.END}\n")
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        print(f"\n{Colors.GREEN}✓ {script_name} completed successfully{Colors.END}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n{Colors.RED}✗ Error running {script_name}{Colors.END}")
        print(f"{Colors.RED}Exit code: {e.returncode}{Colors.END}")
        return False
    except FileNotFoundError:
        print(f"\n{Colors.RED}✗ Script not found: {script_name}{Colors.END}")
        return False

def check_dependencies():
    """Check if required folders exist"""
    print_header("CHECKING SETUP")
    
    folders = ['data', 'models', 'results']
    missing_folders = []
    
    for folder in folders:
        if not os.path.exists(folder):
            missing_folders.append(folder)
            print(f"{Colors.YELLOW}⚠ Creating missing folder: {folder}/{Colors.END}")
            os.makedirs(folder)
        else:
            print(f"{Colors.GREEN}✓ Folder exists: {folder}/{Colors.END}")
    
    if missing_folders:
        print(f"\n{Colors.GREEN}Created {len(missing_folders)} folder(s){Colors.END}")
    
    return True

def main():
    """Main pipeline execution"""
    
    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║                                                               ║")
    print("║        ML TRADING BOT - FULL PIPELINE EXECUTION               ║")
    print("║                                                               ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}\n")
    
    # Check setup
    if not check_dependencies():
        print(f"{Colors.RED}Setup check failed. Exiting.{Colors.END}")
        sys.exit(1)
    
    # Define the pipeline
    pipeline = [
        ("scripts/data_collector.py", "Collecting Historical Data"),
        ("scripts/feature_calc.py", "Engineering Features"),
        ("scripts/train_model.py", "Training ML Model"),
        ("scripts/backtest.py", "Running Backtest"),
    ]
    
    # Track success/failure
    results = []
    
    # Run each script in order
    for script_name, description in pipeline:
        success = run_script(script_name, description)
        results.append((script_name, success))
        
        if not success:
            print(f"\n{Colors.RED}{Colors.BOLD}Pipeline stopped due to error.{Colors.END}")
            print(f"{Colors.RED}Fix the error in {script_name} and try again.{Colors.END}\n")
            sys.exit(1)
    
    # Print summary
    print_header("PIPELINE COMPLETE")
    
    print(f"{Colors.BOLD}Summary:{Colors.END}\n")
    for script_name, success in results:
        status = f"{Colors.GREEN}✓ SUCCESS{Colors.END}" if success else f"{Colors.RED}✗ FAILED{Colors.END}"
        print(f"  {script_name:30s} {status}")
    
    print(f"\n{Colors.GREEN}{Colors.BOLD}All steps completed successfully!{Colors.END}")
    print(f"\n{Colors.CYAN}Next steps:{Colors.END}")
    print(f"  • Check results/backtest_results.png for performance chart")
    print(f"  • Review models/trained_model.pkl")
    print(f"  • Experiment with different features or parameters\n")

if __name__ == "__main__":
    main()