"""
Script to run all case files sequentially
"""
import subprocess
import sys
from pathlib import Path
import time

# Define all case files in execution order
case_files = [
    "case00_es_pv_only.py",
]

def run_case(case_file):
    """Run a single case file"""
    print(f"\n{'='*60}")
    print(f"Starting: {case_file}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, case_file],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✓ Successfully completed: {case_file}")
            print(f"  Execution time: {elapsed_time:.2f} seconds")
            return True
        else:
            print(f"✗ Failed: {case_file}")
            print(f"  Error output:")
            print(result.stderr)
            return False
            
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"✗ Exception in {case_file}: {str(e)}")
        return False

def main():
    """Main execution function"""
    print("Starting batch execution of all cases...")
    print(f"Total cases to run: {len(case_files)}")
    
    total_start = time.time()
    results = {}
    
    for case_file in case_files:
        case_path = Path(__file__).parent / case_file
        
        if not case_path.exists():
            print(f"\n⚠ Warning: {case_file} not found, skipping...")
            results[case_file] = "Not Found"
            continue
        
        success = run_case(case_file)
        results[case_file] = "Success" if success else "Failed"
    
    total_elapsed = time.time() - total_start
    
    # Print summary
    print(f"\n{'='*60}")
    print("EXECUTION SUMMARY")
    print(f"{'='*60}")
    
    for case_file, status in results.items():
        status_symbol = "✓" if status == "Success" else ("✗" if status == "Failed" else "⚠")
        print(f"{status_symbol} {case_file}: {status}")
    
    success_count = sum(1 for s in results.values() if s == "Success")
    failed_count = sum(1 for s in results.values() if s == "Failed")
    skipped_count = sum(1 for s in results.values() if s == "Not Found")
    
    print(f"\nTotal execution time: {total_elapsed:.2f} seconds")
    print(f"Successful: {success_count}/{len(case_files)}")
    print(f"Failed: {failed_count}/{len(case_files)}")
    print(f"Skipped: {skipped_count}/{len(case_files)}")

if __name__ == "__main__":
    main()
