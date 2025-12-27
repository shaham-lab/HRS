"""
Unit tests for load_config module.
"""

import unittest
import json
import os
import tempfile
import shutil
from unittest.mock import patch, mock_open
import sys

# Add P-CAFE directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'P-CAFE'))

from load_config import deep_update, load_hierarchical_config, load_config


class TestDeepUpdate(unittest.TestCase):
    """Test cases for deep_update function."""
    
    def test_deep_update_simple_dict(self):
        """Test updating a simple dictionary."""
        base = {"a": 1, "b": 2}
        update = {"b": 3, "c": 4}
        result = deep_update(base, update)
        
        self.assertEqual(result["a"], 1)
        self.assertEqual(result["b"], 3)
        self.assertEqual(result["c"], 4)
    
    def test_deep_update_nested_dict(self):
        """Test updating nested dictionaries."""
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        update = {"a": {"y": 5, "z": 6}}
        result = deep_update(base, update)
        
        self.assertEqual(result["a"]["x"], 1)
        self.assertEqual(result["a"]["y"], 5)
        self.assertEqual(result["a"]["z"], 6)
        self.assertEqual(result["b"], 3)
    
    def test_deep_update_replace_non_dict(self):
        """Test replacing non-dict values."""
        base = {"a": [1, 2, 3], "b": "old"}
        update = {"a": [4, 5], "b": "new"}
        result = deep_update(base, update)
        
        self.assertEqual(result["a"], [4, 5])
        self.assertEqual(result["b"], "new")
    
    def test_deep_update_empty_update(self):
        """Test updating with empty dictionary."""
        base = {"a": 1, "b": 2}
        update = {}
        result = deep_update(base, update)
        
        self.assertEqual(result, base)
    
    def test_deep_update_empty_base(self):
        """Test updating empty base dictionary."""
        base = {}
        update = {"a": 1, "b": 2}
        result = deep_update(base, update)
        
        self.assertEqual(result, update)
    
    def test_deep_update_deep_nesting(self):
        """Test deeply nested dictionary update."""
        base = {"level1": {"level2": {"level3": {"value": 1}}}}
        update = {"level1": {"level2": {"level3": {"value": 2, "new": 3}}}}
        result = deep_update(base, update)
        
        self.assertEqual(result["level1"]["level2"]["level3"]["value"], 2)
        self.assertEqual(result["level1"]["level2"]["level3"]["new"], 3)


class TestLoadHierarchicalConfig(unittest.TestCase):
    """Test cases for load_hierarchical_config function."""
    
    def setUp(self):
        """Set up temporary directory for test config files."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create base config
        self.base_config = {
            "param1": "base_value1",
            "param2": "base_value2",
            "nested": {"key1": "base_nested1", "key2": "base_nested2"}
        }
        self.base_config_path = os.path.join(self.test_dir, "base_config.json")
        with open(self.base_config_path, 'w') as f:
            json.dump(self.base_config, f)
        
        # Create user config
        self.user_config = {
            "param2": "user_value2",
            "param3": "user_value3",
            "nested": {"key2": "user_nested2", "key3": "user_nested3"}
        }
        self.user_config_path = os.path.join(self.test_dir, "user_config.json")
        with open(self.user_config_path, 'w') as f:
            json.dump(self.user_config, f)
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)
    
    def test_load_base_config_only(self):
        """Test loading base config without user config."""
        # Use non-existent user config
        result = load_hierarchical_config(
            base_config_path=self.base_config_path,
            user_config_path=os.path.join(self.test_dir, "nonexistent.json")
        )
        
        self.assertEqual(result["param1"], "base_value1")
        self.assertEqual(result["param2"], "base_value2")
    
    def test_load_with_user_config(self):
        """Test loading with user config override."""
        result = load_hierarchical_config(
            base_config_path=self.base_config_path,
            user_config_path=self.user_config_path
        )
        
        # Base values
        self.assertEqual(result["param1"], "base_value1")
        # Overridden values
        self.assertEqual(result["param2"], "user_value2")
        # New values from user config
        self.assertEqual(result["param3"], "user_value3")
        # Nested overrides
        self.assertEqual(result["nested"]["key1"], "base_nested1")
        self.assertEqual(result["nested"]["key2"], "user_nested2")
        self.assertEqual(result["nested"]["key3"], "user_nested3")
    
    def test_load_with_cli_args(self):
        """Test loading with CLI argument override."""
        cli_args = {
            "param1": "cli_value1",
            "nested": {"key1": "cli_nested1"}
        }
        
        result = load_hierarchical_config(
            base_config_path=self.base_config_path,
            user_config_path=self.user_config_path,
            cli_args=cli_args
        )
        
        # CLI should override everything
        self.assertEqual(result["param1"], "cli_value1")
        self.assertEqual(result["param2"], "user_value2")
        self.assertEqual(result["nested"]["key1"], "cli_nested1")
        self.assertEqual(result["nested"]["key2"], "user_nested2")
    
    def test_load_missing_base_config(self):
        """Test error when base config is missing."""
        with self.assertRaises(FileNotFoundError):
            load_hierarchical_config(
                base_config_path=os.path.join(self.test_dir, "missing.json"),
                user_config_path=self.user_config_path
            )


class TestLoadConfig(unittest.TestCase):
    """Test cases for legacy load_config function."""
    
    def setUp(self):
        """Set up temporary directory for test config files."""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        # Create base config
        self.base_config = {"param1": "base", "param2": "base"}
        with open("base_config.json", 'w') as f:
            json.dump(self.base_config, f)
    
    def tearDown(self):
        """Clean up temporary directory."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)
    
    def test_load_config_base_only(self):
        """Test loading config with no user-specific config."""
        result = load_config("testuser")
        
        self.assertEqual(result["param1"], "base")
        self.assertEqual(result["param2"], "base")
    
    def test_load_config_with_user_file(self):
        """Test loading config with user-specific config file."""
        # Create user-specific config
        user_config = {"param2": "user_override", "param3": "user_new"}
        with open("user_config_testuser.json", 'w') as f:
            json.dump(user_config, f)
        
        result = load_config("testuser")
        
        self.assertEqual(result["param1"], "base")
        self.assertEqual(result["param2"], "user_override")
        self.assertEqual(result["param3"], "user_new")


if __name__ == '__main__':
    unittest.main()
