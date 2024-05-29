import os
import subprocess
import sys


def calculate_storage(directory):
    if not os.path.isdir(directory):
        print(f"Error: Directory {directory} does not exist.")
        sys.exit(1)

    try:
        result = subprocess.run(
            ['du', '-sh', directory], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print(f"Error calculating storage: {result.stderr.strip()}")
            sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":

    directory = 'maps'
    calculate_storage(directory)
