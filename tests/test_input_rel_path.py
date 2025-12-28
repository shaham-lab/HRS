"""
Unit tests for input_rel_path parameter functionality.
"""

import unittest
import sys
import os
import ast
import inspect


class TestInputRelPathParameter(unittest.TestCase):
    """Test cases for input_rel_path parameter handling through static analysis."""
    
    def test_guesser_main_has_input_rel_path_argument(self):
        """Test that guesser_main.py adds input_rel_path argument to parser."""
        file_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'Guesser', 'guesser_main.py')
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check that the argument is added
        self.assertIn('parser.add_argument("--input_rel_path"', content)
        # Check that it reads from config
        self.assertIn('config.get("input_rel_path"', content)
    
    def test_multimodal_guesser_passes_input_rel_path(self):
        """Test that multimodal_guesser.py passes input_rel_path to data loader."""
        file_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'Guesser', 'multimodal_guesser.py')
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check that load_data_function accepts input_rel_path
        self.assertIn('def load_data_function(data_loader_name, input_rel_path=', content)
        # Check that it's passed to the loader function
        self.assertIn('loader_func(input_rel_path=input_rel_path)', content)
        # Check that FLAGS.input_rel_path is used
        self.assertIn('FLAGS.input_rel_path', content)
    
    def test_utils_load_mimic_time_series_signature(self):
        """Test that load_mimic_time_series has input_rel_path parameter."""
        file_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'common', 'utils.py')
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check function signature
        self.assertIn('def load_mimic_time_series(input_rel_path=', content)
        # Check that it's used in path construction
        self.assertIn('os.path.join(base_dir, input_rel_path,', content)
    
    def test_utils_load_time_Series_signature(self):
        """Test that load_time_Series has input_rel_path parameter."""
        file_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'common', 'utils.py')
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check function signature
        self.assertIn('def load_time_Series(input_rel_path=', content)
        # Check that it's used in path construction with df_data.csv
        lines = content.split('\n')
        found_function = False
        found_usage = False
        for i, line in enumerate(lines):
            if 'def load_time_Series(input_rel_path=' in line:
                found_function = True
            if found_function and 'os.path.join(base_dir, input_rel_path,' in line:
                found_usage = True
                break
        
        self.assertTrue(found_function, "load_time_Series function not found")
        self.assertTrue(found_usage, "input_rel_path not used in path construction")
    
    def test_utils_load_mimic_text_signature(self):
        """Test that load_mimic_text has input_rel_path parameter."""
        file_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'common', 'utils.py')
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check function signature
        self.assertIn('def load_mimic_text(input_rel_path=', content)
        # Check that it's used in path construction with data_with_text.json
        lines = content.split('\n')
        found_function = False
        found_usage = False
        for i, line in enumerate(lines):
            if 'def load_mimic_text(input_rel_path=' in line:
                found_function = True
            if found_function and 'os.path.join(base_dir, input_rel_path,' in line:
                found_usage = True
                break
        
        self.assertTrue(found_function, "load_mimic_text function not found")
        self.assertTrue(found_usage, "input_rel_path not used in path construction")
    
    def test_default_values_match_config(self):
        """Test that default values in functions match config file."""
        # Read config
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'base_config.json')
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        config_value = config.get('input_rel_path', '')
        
        # Read utils.py
        utils_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'common', 'utils.py')
        with open(utils_path, 'r') as f:
            content = f.read()
        
        # Check that default values match
        import re
        defaults = re.findall(r'def load_\w+\(input_rel_path="([^"]+)"\)', content)
        
        for default in defaults:
            # Handle escaped backslashes in Python string literals
            # The source code shows "\\\\" but it represents a single backslash
            python_value = default.encode().decode('unicode_escape')
            self.assertEqual(python_value, config_value, 
                           f"Default value '{python_value}' doesn't match config value '{config_value}'")


if __name__ == '__main__':
    unittest.main()
