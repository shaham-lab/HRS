"""
Unit tests for main_robust configuration loading and argument parsing.
This test validates the structure and patterns used in main_robust.py
"""

import unittest
import json
import os
import tempfile
import shutil
import re
from pathlib import Path


class TestMainRobustStructure(unittest.TestCase):
    """Test that main_robust.py follows the required patterns."""
    
    def setUp(self):
        """Load the main_robust.py file content."""
        self.main_robust_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'src', 
            'DDQN', 
            'main_robust.py'
        )
        with open(self.main_robust_path, 'r') as f:
            self.content = f.read()
    
    def test_has_default_constants(self):
        """Test that hardcoded default constants are defined."""
        required_constants = [
            'DEFAULT_SAVE_DIR',
            'DEFAULT_GAMMA',
            'DEFAULT_BATCH_SIZE',
            'DEFAULT_HIDDEN_DIM',
            'DEFAULT_VAL_INTERVAL',
            'DEFAULT_VAL_TRIALS_WO_IM',
            'DEFAULT_COST_BUDGET'
        ]
        
        for const in required_constants:
            self.assertIn(const, self.content, 
                         f"Missing default constant: {const}")
    
    def test_has_parse_arguments_function(self):
        """Test that parse_arguments function exists."""
        self.assertIn('def parse_arguments():', self.content,
                     "Missing parse_arguments() function")
    
    def test_parse_arguments_uses_hierarchical_config(self):
        """Test that parse_arguments uses load_hierarchical_config."""
        # Check for import
        self.assertIn('from ..common.load_config import load_hierarchical_config', 
                     self.content,
                     "Missing import of load_hierarchical_config")
        
        # Check for usage in parse_arguments
        parse_args_match = re.search(
            r'def parse_arguments\(\):.*?(?=\ndef |\nclass |\Z)',
            self.content,
            re.DOTALL
        )
        self.assertIsNotNone(parse_args_match, "Could not find parse_arguments function")
        
        parse_args_content = parse_args_match.group(0)
        self.assertIn('load_hierarchical_config', parse_args_content,
                     "parse_arguments doesn't call load_hierarchical_config")
        self.assertIn('config/base_config.json', parse_args_content,
                     "parse_arguments doesn't reference base_config.json")
        self.assertIn('config/user_config.json', parse_args_content,
                     "parse_arguments doesn't reference user_config.json")
    
    def test_parse_arguments_uses_config_with_defaults(self):
        """Test that arguments use config values with hardcoded defaults."""
        parse_args_match = re.search(
            r'def parse_arguments\(\):.*?(?=\ndef |\nclass |\Z)',
            self.content,
            re.DOTALL
        )
        parse_args_content = parse_args_match.group(0)
        
        # Check that config.get() is used with default constants
        self.assertIn('main_robust_config.get(', parse_args_content,
                     "parse_arguments doesn't use main_robust_config.get()")
        self.assertIn('DEFAULT_', parse_args_content,
                     "parse_arguments doesn't use DEFAULT_ constants as fallbacks")
    
    def test_no_module_level_config_loading(self):
        """Test that config is not loaded at module level."""
        # Find where parse_arguments is defined
        parse_args_pos = self.content.find('def parse_arguments():')
        
        # Everything before parse_arguments should not have json.load or config loading
        before_parse_args = self.content[:parse_args_pos]
        
        # Should not have json.load at module level
        self.assertNotIn("with open(r'config", before_parse_args,
                        "Config should not be loaded at module level")
        self.assertNotIn("json.load(f)", before_parse_args,
                        "json.load should not be called at module level")
    
    def test_no_module_level_parser_creation(self):
        """Test that argument parser is not created at module level."""
        parse_args_pos = self.content.find('def parse_arguments():')
        before_parse_args = self.content[:parse_args_pos]
        
        # Should not have parser = argparse.ArgumentParser at module level
        # (outside of parse_arguments function)
        self.assertNotRegex(before_parse_args, 
                           r'^parser\s*=\s*argparse\.ArgumentParser',
                           "ArgumentParser should not be created at module level")
    
    def test_no_module_level_flags(self):
        """Test that FLAGS is not defined at module level."""
        parse_args_pos = self.content.find('def parse_arguments():')
        before_parse_args = self.content[:parse_args_pos]
        
        # Should not have FLAGS = parser.parse_args at module level
        self.assertNotIn('FLAGS = parser.parse_args', before_parse_args,
                        "FLAGS should not be defined at module level")
    
    def test_run_function_accepts_flags_parameter(self):
        """Test that run function accepts FLAGS as parameter."""
        # Look for the run function definition
        run_match = re.search(r'def run\([^)]*\):', self.content)
        self.assertIsNotNone(run_match, "Could not find run() function")
        
        # Check that it accepts FLAGS parameter
        self.assertIn('FLAGS', run_match.group(0),
                     "run() function should accept FLAGS as parameter")
    
    def test_main_calls_parse_arguments(self):
        """Test that main() calls parse_arguments()."""
        # Find main function
        main_match = re.search(
            r'def main\(\):.*?(?=\ndef |\nif __name__ |\Z)',
            self.content,
            re.DOTALL
        )
        self.assertIsNotNone(main_match, "Could not find main() function")
        
        main_content = main_match.group(0)
        self.assertIn('parse_arguments()', main_content,
                     "main() should call parse_arguments()")
        self.assertIn('FLAGS =', main_content,
                     "main() should assign result of parse_arguments() to FLAGS")
    
    def test_main_passes_flags_to_run(self):
        """Test that main() passes FLAGS to run()."""
        main_match = re.search(
            r'def main\(\):.*?(?=\ndef |\nif __name__ |\Z)',
            self.content,
            re.DOTALL
        )
        main_content = main_match.group(0)
        
        self.assertIn('run(FLAGS)', main_content,
                     "main() should call run(FLAGS)")
    
    def test_functions_receive_specific_parameters(self):
        """Test that key functions receive specific parameters instead of using FLAGS."""
        # Check train_helper
        self.assertRegex(self.content, 
                        r'def train_helper\([^)]*gamma[^)]*device[^)]*\)',
                        "train_helper should accept gamma and device as parameters")
        
        # Check play_episode
        self.assertRegex(self.content,
                        r'def play_episode\([^)]*gamma[^)]*device[^)]*\)',
                        "play_episode should accept gamma and device as parameters")
        
        # Check save_networks
        self.assertRegex(self.content,
                        r'def save_networks\([^)]*save_dir[^)]*device[^)]*\)',
                        "save_networks should accept save_dir and device as parameters")


if __name__ == '__main__':
    unittest.main()

