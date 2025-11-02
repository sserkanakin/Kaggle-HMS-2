#!/usr/bin/env python3
"""
Quick diagnostic to recommend optimal worker count based on system resources
"""

import os
import psutil
import multiprocessing

def get_optimal_workers():
    """Recommend optimal worker count based on available RAM"""
    
    # Get system info
    cpu_count = multiprocessing.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    print("=" * 60)
    print("System Resources")
    print("=" * 60)
    print(f"CPU Cores: {cpu_count}")
    print(f"Total RAM: {memory_gb:.1f} GB")
    print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    
    # Estimate memory per worker
    # Each worker can use ~2-4 GB for graph processing
    memory_per_worker_gb = 3.0  # Conservative estimate
    
    # Calculate max workers based on memory
    max_workers_by_memory = int(memory_gb * 0.7 / memory_per_worker_gb)
    
    # Recommendation
    recommended = min(4, max_workers_by_memory, cpu_count - 1)
    recommended = max(1, recommended)  # At least 1
    
    print("\n" + "=" * 60)
    print("Worker Recommendations")
    print("=" * 60)
    print(f"Maximum by CPU: {cpu_count - 1}")
    print(f"Maximum by Memory (70% usage): {max_workers_by_memory}")
    print(f"\n✅ RECOMMENDED: {recommended} workers")
    
    if memory_gb < 16:
        print("\n⚠️  WARNING: Less than 16GB RAM detected")
        print("   Consider using 2-3 workers maximum")
    
    print("\nUsage:")
    print(f"  python src/data/make_graph_dataset.py --workers {recommended}")
    print("\nFor debugging (if errors persist):")
    print(f"  python src/data/make_graph_dataset.py --workers 1")
    print("=" * 60)

if __name__ == "__main__":
    get_optimal_workers()
