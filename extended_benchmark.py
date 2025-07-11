#!/usr/bin/env python3
"""
Extended benchmark for longer strings where Ukkonen should excel
"""

from htypist import EditDistanceOptimizer
import time

def extended_benchmark():
    """Test with longer strings where Ukkonen's algorithm should excel"""
    print("Extended Ukkonen's Algorithm Benchmark")
    print("=" * 50)

    # Longer, more realistic code snippets
    test_cases = [
        # Medium length with few errors (Ukkonen's sweet spot)
        ('import torch\nimport torch.nn as nn\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()',
         'import torch\nimport torch.nn as nn\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()'),

        # Longer string with 1 typo
        ('def train_model(model, dataloader, optimizer, epochs=10):\n    model.train()\n    for epoch in range(epochs):',
         'def train_model(model, dataloader, optimizer, epochs=10):\n    model.train()\n    for epoch in rang(epochs):'),

        # Very long string with perfect match (best case for early termination)
        ('from sklearn.model_selection import train_test_split\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import accuracy_score\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)',
         'from sklearn.model_selection import train_test_split\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import accuracy_score\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)'),

        # Long string with 1 error
        ('import pandas as pd\nimport numpy as np\ndf = pd.read_csv("data.csv")\ndf_cleaned = df.dropna()\ncorrelation_matrix = df.corr()',
         'import pandas as pd\nimport numpy as np\ndf = pd.read_csv("data.csv")\ndf_cleaned = df.dropna()\ncorrelation_matrix = df.cor()'),
    ]

    total_speedup = 0.0

    for i, (s1, s2) in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"String lengths: {len(s1)} vs {len(s2)} characters")

        # Test correctness
        ukkonen_result = EditDistanceOptimizer.ukkonen_edit_distance(s1, s2)
        standard_result = EditDistanceOptimizer.calculate_edit_distance(s1, s2, use_optimization=False)

        print(f"Edit distance: {ukkonen_result}")
        assert ukkonen_result == standard_result, f"Results don't match! Ukkonen: {ukkonen_result}, Standard: {standard_result}"

        # Benchmark with fewer iterations for longer strings
        iterations = 500

        # Time Ukkonen
        start = time.perf_counter()
        for _ in range(iterations):
            EditDistanceOptimizer.ukkonen_edit_distance(s1, s2)
        ukkonen_time = time.perf_counter() - start

        # Time standard
        start = time.perf_counter()
        for _ in range(iterations):
            EditDistanceOptimizer.calculate_edit_distance(s1, s2, use_optimization=False)
        standard_time = time.perf_counter() - start

        speedup = standard_time / ukkonen_time if ukkonen_time > 0 else 1.0
        total_speedup += speedup

        print(f"Ukkonen:  {ukkonen_time*1000:.3f}ms ({iterations} iterations)")
        print(f"Standard: {standard_time*1000:.3f}ms ({iterations} iterations)")
        print(f"Speedup:  {speedup:.2f}x")

        # Show theoretical complexity benefit
        complexity_ratio = (len(s1) * len(s2)) / (min(len(s1), len(s2)) * max(1, ukkonen_result))
        print(f"Theoretical complexity benefit: {complexity_ratio:.1f}x")

    avg_speedup = total_speedup / len(test_cases)
    print("\n" + "=" * 50)
    print(f"Average speedup: {avg_speedup:.2f}x")

    if avg_speedup > 2.0:
        print("🚀 Excellent performance improvement!")
    elif avg_speedup > 1.5:
        print("✅ Good performance improvement!")
    elif avg_speedup > 1.1:
        print("👍 Moderate performance improvement.")
    else:
        print("📊 Modest improvement (expected for small strings/distances).")

    print("\nKey insights:")
    print("- Ukkonen's algorithm excels with longer strings and small edit distances")
    print("- For very short strings, overhead may outweigh benefits")
    print("- Real-world typing scenarios typically have few errors (Ukkonen's strength)")

    return avg_speedup

if __name__ == "__main__":
    extended_benchmark()
