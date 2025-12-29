"""
Unit tests for configuration loading and argument parsing.
This test validates the structure and patterns used in parse_args.py and its usage.
"""

import unittest
import json
import os
import tempfile
import shutil
import re
from pathlib import Path


class TestParseArgsStructure(unittest.TestCase):
    """Test that parse_args.py follows the required patterns."""
    
    def setUp(self):
        """Load the parse_args.py file content."""
        self.parse_args_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'src', 
            'common', 
            'parse_args.py'
        )
        with open(self.parse_args_path, 'r') as f:
            self.content = f.read()
    
    def test_has_default_constants(self):
        """Test that hardcoded default constants are defined in parse_args.py."""
        required_constants = [
            'DEFAULT_GAMMA',
            'DEFAULT_BATCH_SIZE',
            'DEFAULT_HIDDEN_DIM',
            'DEFAULT_VAL_INTERVAL',
            'DEFAULT_VAL_TRIALS_WO_IM',
            'DEFAULT_COST_BUDGET',
            'DEFAULT_NUM_EPOCHS',
            'DEFAULT_VAL_TRIALS_WO_IM_GUESSER'
        ]
        
        for const in required_constants:
            self.assertIn(const, self.content, 
                         f"Missing default constant: {const}")
    
    def test_has_parse_arguments_function(self):
        """Test that parse_arguments function exists."""
        self.assertIn('def parse_arguments():', self.content,
                     "Missing parse_arguments() function")
    
    def test_has_parse_embedder_guesser_args(self):
        """Test that parse_embedder_guesser_args function exists."""
        self.assertIn('def parse_embedder_guesser_args(', self.content,
                     "Missing parse_embedder_guesser_args() function")
    
    def test_has_parse_main_robust_args(self):
        """Test that parse_main_robust_args function exists."""
        self.assertIn('def parse_main_robust_args(', self.content,
                     "Missing parse_main_robust_args() function")
    
    def test_parse_arguments_uses_hierarchical_config(self):
        """Test that parse_arguments uses load_hierarchical_config."""
        # Check for import
        self.assertIn('from .load_config import load_hierarchical_config', 
                     self.content,
                     "Missing import of load_hierarchical_config")
        
        # Check for usage in parse_arguments
        parse_args_match = re.search(
            r'def parse_arguments\(\):.*?(?=\ndef |\Z)',
            self.content,
            re.DOTALL
        )
        self.assertIsNotNone(parse_args_match, "Could not find parse_arguments function")
        
        parse_args_content = parse_args_match.group(0)
        self.assertIn('load_hierarchical_config', parse_args_content,
                     "parse_arguments doesn't call load_hierarchical_config")
    
    def test_parse_arguments_calls_both_functions(self):
        """Test that parse_arguments calls both parse_embedder_guesser_args and parse_main_robust_args."""
        parse_args_match = re.search(
            r'def parse_arguments\(\):.*?(?=\ndef |\Z)',
            self.content,
            re.DOTALL
        )
        parse_args_content = parse_args_match.group(0)
        
        self.assertIn('parse_embedder_guesser_args', parse_args_content,
                     "parse_arguments doesn't call parse_embedder_guesser_args")
        self.assertIn('parse_main_robust_args', parse_args_content,
                     "parse_arguments doesn't call parse_main_robust_args")


class TestMainRobustStructure(unittest.TestCase):
    """Test that main_robust.py uses the new parse_arguments from parse_args."""
    
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
    
    def test_imports_parse_arguments_from_parse_args(self):
        """Test that main_robust imports parse_arguments from parse_args."""
        self.assertIn('from ..common.parse_args import parse_arguments', 
                     self.content,
                     "Missing import of parse_arguments from parse_args")
    
    def test_no_local_parse_arguments(self):
        """Test that main_robust doesn't define its own parse_arguments."""
        # Should not have a local definition of parse_arguments
        self.assertNotIn('def parse_arguments():', self.content,
                        "main_robust should not define its own parse_arguments")
    
    def test_no_default_constants(self):
        """Test that main_robust doesn't have default constants (they're in parse_args now)."""
        # Should not have DEFAULT_ constants in main_robust
        # Allow DEVICE since that's a module-level constant
        default_pattern = r'DEFAULT_(?!.*DEVICE)[A-Z_]+'
        matches = re.findall(default_pattern, self.content)
        # Filter out any in comments or strings
        actual_definitions = [m for m in matches if f'{m} =' in self.content]
        self.assertEqual(len(actual_definitions), 0,
                        f"main_robust should not define DEFAULT constants, found: {actual_definitions}")
    
    def test_has_device_global_constant(self):
        """Test that DEVICE is defined as a global constant."""
        self.assertIn('DEVICE = torch.device(', self.content,
                     "DEVICE should be defined as a module-level constant")
    
    def test_no_flags_stub(self):
        """Test that flags_stub is not created in load_networks."""
        self.assertNotIn('flags_stub', self.content,
                        "flags_stub should not be used; FLAGS should be passed instead")
    
    def test_load_networks_accepts_flags(self):
        """Test that load_networks accepts FLAGS as a parameter."""
        self.assertRegex(self.content,
                        r'def load_networks\([^)]*FLAGS[^)]*\)',
                        "load_networks should accept FLAGS as parameter")
    
    def test_main_calls_parse_arguments(self):
        """Test that main() calls parse_arguments()."""
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
    
    def test_run_function_accepts_flags_parameter(self):
        """Test that run function accepts FLAGS as parameter."""
        run_match = re.search(r'def run\([^)]*\):', self.content)
        self.assertIsNotNone(run_match, "Could not find run() function")
        
        self.assertIn('FLAGS', run_match.group(0),
                     "run() function should accept FLAGS as parameter")


class TestGuesserMainStructure(unittest.TestCase):
    """Test that guesser_main.py uses the new parse_arguments from parse_args."""
    
    def setUp(self):
        """Load the guesser_main.py file content."""
        self.guesser_main_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'src', 
            'Guesser', 
            'guesser_main.py'
        )
        with open(self.guesser_main_path, 'r') as f:
            self.content = f.read()
    
    def test_imports_parse_arguments_from_parse_args(self):
        """Test that guesser_main imports parse_arguments from parse_args."""
        self.assertIn('from ..common.parse_args import parse_arguments', 
                     self.content,
                     "Missing import of parse_arguments from parse_args")
    
    def test_no_local_parse_arguments(self):
        """Test that guesser_main doesn't define its own parse_arguments."""
        self.assertNotIn('def parse_arguments():', self.content,
                        "guesser_main should not define its own parse_arguments")
    
    def test_no_default_constants(self):
        """Test that guesser_main doesn't have default constants (they're in parse_args now)."""
        # Should not have NUM_EPOCHS or VAL_TRIALS_WO_IM constants
        self.assertNotIn('NUM_EPOCHS =', self.content,
                        "guesser_main should not define NUM_EPOCHS")
        self.assertNotIn('VAL_TRIALS_WO_IM =', self.content,
                        "guesser_main should not define VAL_TRIALS_WO_IM")


if __name__ == '__main__':
    unittest.main()

