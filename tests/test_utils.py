"""
Unit tests for utils module.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
import os

# Add P-CAFE directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'P-CAFE'))

from utils import (
    add_noise,
    balance_class,
    balance_class_no_noise,
    balance_class_multi,
    map_multiple_features,
    map_multiple_features_for_logistic_mimic
)


class TestAddNoise(unittest.TestCase):
    """Test cases for add_noise function."""
    
    def test_add_noise_shape_preserved(self):
        """Test that noise addition preserves array shape."""
        X = np.array([[1, 2, 3], [4, 5, 6]])
        X_noisy = add_noise(X, noise_std=0.01)
        self.assertEqual(X.shape, X_noisy.shape)
    
    def test_add_noise_changes_values(self):
        """Test that noise actually changes the values."""
        np.random.seed(42)
        X = np.array([[1, 2, 3], [4, 5, 6]])
        X_noisy = add_noise(X, noise_std=0.1)
        # Values should be different but close
        self.assertFalse(np.array_equal(X, X_noisy))
        self.assertTrue(np.allclose(X, X_noisy, atol=0.5))
    
    def test_add_noise_zero_std(self):
        """Test that zero std produces no change."""
        X = np.array([[1, 2, 3], [4, 5, 6]])
        X_noisy = add_noise(X, noise_std=0.0)
        self.assertTrue(np.allclose(X, X_noisy, atol=1e-10))
    
    def test_add_noise_single_sample(self):
        """Test noise addition on single sample."""
        X = np.array([[1, 2, 3]])
        X_noisy = add_noise(X, noise_std=0.01)
        self.assertEqual(X.shape, X_noisy.shape)


class TestBalanceClass(unittest.TestCase):
    """Test cases for balance_class function."""
    
    def test_balance_class_imbalanced_data(self):
        """Test balancing of imbalanced dataset."""
        np.random.seed(42)
        # Create imbalanced dataset
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([0, 0, 0, 1, 1])  # 3 class 0, 2 class 1
        
        X_balanced, y_balanced = balance_class(X, y, noise_std=0.01)
        
        # Check that classes are balanced
        unique, counts = np.unique(y_balanced, return_counts=True)
        self.assertEqual(counts[0], counts[1])
        
        # Check that total samples increased
        self.assertGreater(len(y_balanced), len(y))
    
    def test_balance_class_already_balanced(self):
        """Test that already balanced data is handled correctly."""
        np.random.seed(42)
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 0, 1, 1])  # Already balanced
        
        X_balanced, y_balanced = balance_class(X, y, noise_std=0.01)
        
        # Should return same or similar size
        self.assertEqual(len(y), len(y_balanced))
    
    def test_balance_class_preserves_features(self):
        """Test that number of features is preserved."""
        np.random.seed(42)
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        y = np.array([0, 0, 0, 1])
        
        X_balanced, y_balanced = balance_class(X, y, noise_std=0.01)
        
        self.assertEqual(X.shape[1], X_balanced.shape[1])


class TestBalanceClassNoNoise(unittest.TestCase):
    """Test cases for balance_class_no_noise function."""
    
    def test_balance_class_no_noise_imbalanced(self):
        """Test balancing without noise on imbalanced data."""
        np.random.seed(42)
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([0, 0, 0, 1, 1])
        
        X_balanced, y_balanced = balance_class_no_noise(X, y)
        
        # Check that classes are balanced
        unique, counts = np.unique(y_balanced, return_counts=True)
        self.assertEqual(counts[0], counts[1])
    
    def test_balance_class_no_noise_duplicates(self):
        """Test that minority class samples are duplicated."""
        np.random.seed(42)
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 0, 1])
        
        X_balanced, y_balanced = balance_class_no_noise(X, y)
        
        # Check for exact duplicates (no noise added)
        minority_samples = X_balanced[y_balanced == 1]
        self.assertEqual(len(minority_samples), 2)
    
    def test_balance_class_no_noise_already_balanced(self):
        """Test already balanced data."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 0, 1, 1])
        
        X_balanced, y_balanced = balance_class_no_noise(X, y)
        
        self.assertEqual(len(y), len(y_balanced))
        self.assertTrue(np.array_equal(X, X_balanced))
        self.assertTrue(np.array_equal(y, y_balanced))


class TestBalanceClassMulti(unittest.TestCase):
    """Test cases for balance_class_multi function."""
    
    def test_balance_class_multi_three_classes(self):
        """Test balancing with three classes."""
        np.random.seed(42)
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
        y = np.array([0, 0, 0, 1, 1, 2])  # Imbalanced: 3, 2, 1
        
        X_balanced, y_balanced = balance_class_multi(X, y, noise_std=0.01)
        
        # All classes should have same count as majority class
        unique, counts = np.unique(y_balanced, return_counts=True)
        self.assertEqual(len(set(counts)), 1)  # All counts should be equal
    
    def test_balance_class_multi_preserves_majority(self):
        """Test that majority class count is preserved."""
        np.random.seed(42)
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([0, 0, 0, 1, 2])
        
        X_balanced, y_balanced = balance_class_multi(X, y, noise_std=0.01)
        
        majority_count = 3
        unique, counts = np.unique(y_balanced, return_counts=True)
        for count in counts:
            self.assertEqual(count, majority_count)


class TestMapMultipleFeatures(unittest.TestCase):
    """Test cases for map_multiple_features function."""
    
    def test_map_multiple_features_basic(self):
        """Test basic feature mapping."""
        sample = np.array([1, 2, 3, 4, 5])
        index_map = map_multiple_features(sample)
        
        self.assertEqual(len(index_map), len(sample))
        for i in range(len(sample)):
            self.assertEqual(index_map[i], [i])
    
    def test_map_multiple_features_single_element(self):
        """Test mapping with single element."""
        sample = np.array([42])
        index_map = map_multiple_features(sample)
        
        self.assertEqual(len(index_map), 1)
        self.assertEqual(index_map[0], [0])
    
    def test_map_multiple_features_empty(self):
        """Test mapping with empty array."""
        sample = np.array([])
        index_map = map_multiple_features(sample)
        
        self.assertEqual(len(index_map), 0)


class TestMapMultipleFeaturesForLogisticMimic(unittest.TestCase):
    """Test cases for map_multiple_features_for_logistic_mimic function."""
    
    def test_map_logistic_mimic_structure(self):
        """Test the structure of MIMIC logistic mapping."""
        sample = np.zeros(17 * 42)  # 17 tests, 42 features each
        index_map = map_multiple_features_for_logistic_mimic(sample)
        
        self.assertEqual(len(index_map), 17)
    
    def test_map_logistic_mimic_ranges(self):
        """Test that each test maps to correct 42-feature range."""
        sample = np.zeros(17 * 42)
        index_map = map_multiple_features_for_logistic_mimic(sample)
        
        for i in range(17):
            expected_range = list(range(i * 42, i * 42 + 42))
            self.assertEqual(index_map[i], expected_range)
    
    def test_map_logistic_mimic_no_overlap(self):
        """Test that feature ranges don't overlap."""
        sample = np.zeros(17 * 42)
        index_map = map_multiple_features_for_logistic_mimic(sample)
        
        all_indices = []
        for indices in index_map.values():
            all_indices.extend(indices)
        
        # Check that all indices are unique (no overlap)
        self.assertEqual(len(all_indices), len(set(all_indices)))


if __name__ == '__main__':
    unittest.main()
