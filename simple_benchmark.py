#!/usr/bin/env python3
"""
Simple benchmark test for Ukkonen's algorithm
"""

from htypist import EditDistanceOptimizer
import time

def simple_benchmark():
    """Run a simple performance comparison"""
    print("Simple Ukkonen's Algorithm Benchmark")
    print("=" * 45)

    test_cases = [
        ('print("Hello")', 'pritn("Hello")'),  # 1 error
        ('def func():', 'def func():'),  # 0 errors
        ('import numpy', 'import numpi'),  # 1 error
        ('for i in range(10):', 'for i in rang(10):'),  # 1 error
    ]

    total_speedup = 0.0

    for i, (s1, s2) in enumerate(test_cases, 1):
        print(f"\nTest {i}: '{s1}' vs '{s2}'")

        # Test correctness first
        ukkonen_result = EditDistanceOptimizer.ukkonen_edit_distance(s1, s2)
        standard_result = EditDistanceOptimizer.calculate_edit_distance(s1, s2, use_optimization=False)

        print(f"Edit distance: {ukkonen_result}")
        assert ukkonen_result == standard_result, "Results don't match!"

        # Time Ukkonen
        start = time.perf_counter()
        for _ in range(1000):
            EditDistanceOptimizer.ukkonen_edit_distance(s1, s2)
        ukkonen_time = time.perf_counter() - start

        # Time standard
        start = time.perf_counter()
        for _ in range(1000):
            EditDistanceOptimizer.calculate_edit_distance(s1, s2, use_optimization=False)
        standard_time = time.perf_counter() - start

        speedup = standard_time / ukkonen_time if ukkonen_time > 0 else 1.0
        total_speedup += speedup

        print(f"Ukkonen:  {ukkonen_time*1000:.3f}ms")
        print(f"Standard: {standard_time*1000:.3f}ms")
        print(f"Speedup:  {speedup:.2f}x")

    avg_speedup = total_speedup / len(test_cases)
    print(f"\nAverage speedup: {avg_speedup:.2f}x")

    if avg_speedup > 1.2:
        print("✅ Good performance improvement!")
    else:
        print("📊 Modest performance improvement.")

    return avg_speedup

if __name__ == "__main__":
    simple_benchmark()
