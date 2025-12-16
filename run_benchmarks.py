import subprocess
import os

def run_command(command):
    print(f"Running: {command}")
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(e)

def main():
    algorithms = [
        "AssignmentPysparkMSTfordensegraphs.py",
        "PysparkMSTfordensegraphsfast.py"
    ]
    
    epsilons = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

    for script in algorithms:
        print(f"\n========== Benchmarking {script} ==========")
        
        for eps in epsilons:
            print(f"\n--- Running with epsilon={eps} ---")
            
            cmd = f"python3 {script} --epsilon {eps}"
            run_command(cmd)

if __name__ == "__main__":
    main()
