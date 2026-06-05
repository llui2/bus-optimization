from __future__ import annotations

import os
import subprocess
import sys


def main() -> None:
    base_dir = os.path.dirname(__file__)
    tests_dir = os.path.join(base_dir, "tests")

    test_files = [
        "test_network.py",
        "test_line_builder.py",
        "test_objective.py",
        "test_ga.py",
    ]

    all_ok = True

    for test_file in test_files:
        test_path = os.path.join(tests_dir, test_file)

        print(f"\n=== Executant {test_file} ===")
        result = subprocess.run([sys.executable, test_path])

        if result.returncode != 0:
            print(f"\nERROR: ha fallat {test_file}")
            all_ok = False
            break

    if all_ok:
        print("\nTots els tests han acabat correctament.")
    else:
        print("\nAlguns tests han fallat.")


if __name__ == "__main__":
    main()
