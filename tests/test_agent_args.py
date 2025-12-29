"""
Unit tests for agent argument parsing.
This test validates that agent arguments are properly integrated into parse_args.py.
"""

import unittest
import re
import os


class TestAgentArgsStructure(unittest.TestCase):
    """Test that agent arguments are properly defined in parse_args.py."""
    
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
    
    def test_has_agent_default_constants(self):
        """Test that agent default constants are defined in parse_args.py."""
        required_constants = [
            'DEFAULT_DECAY_STEP_SIZE',
            'DEFAULT_MIN_LR'
        ]
        
        for const in required_constants:
            self.assertIn(const, self.content, 
                         f"Missing default constant: {const}")
    
    def test_has_parse_agent_args_function(self):
        """Test that parse_agent_args function exists."""
        self.assertIn('def parse_agent_args(', self.content,
                     "Missing parse_agent_args() function")
    
    def test_parse_arguments_calls_parse_agent_args(self):
        """Test that parse_arguments calls parse_agent_args."""
        parse_args_match = re.search(
            r'def parse_arguments\(\):.*?(?=\ndef |\Z)',
            self.content,
            re.DOTALL
        )
        self.assertIsNotNone(parse_args_match, "Could not find parse_arguments function")
        
        parse_args_content = parse_args_match.group(0)
        self.assertIn('parse_agent_args', parse_args_content,
                     "parse_arguments doesn't call parse_agent_args")
    
    def test_parse_agent_args_adds_decay_step_size(self):
        """Test that parse_agent_args adds decay_step_size argument."""
        parse_agent_match = re.search(
            r'def parse_agent_args\(.*?\):.*?(?=\ndef |\Z)',
            self.content,
            re.DOTALL
        )
        self.assertIsNotNone(parse_agent_match, "Could not find parse_agent_args function")
        
        parse_agent_content = parse_agent_match.group(0)
        self.assertIn('--decay_step_size', parse_agent_content,
                     "parse_agent_args doesn't add --decay_step_size argument")
    
    def test_parse_agent_args_adds_min_lr(self):
        """Test that parse_agent_args adds min_lr argument."""
        parse_agent_match = re.search(
            r'def parse_agent_args\(.*?\):.*?(?=\ndef |\Z)',
            self.content,
            re.DOTALL
        )
        self.assertIsNotNone(parse_agent_match, "Could not find parse_agent_args function")
        
        parse_agent_content = parse_agent_match.group(0)
        self.assertIn('--min_lr', parse_agent_content,
                     "parse_agent_args doesn't add --min_lr argument")
    
    def test_parse_agent_args_handles_lr_decay_factor(self):
        """Test that parse_agent_args handles lr_decay_factor argument."""
        parse_agent_match = re.search(
            r'def parse_agent_args\(.*?\):.*?(?=\ndef |\Z)',
            self.content,
            re.DOTALL
        )
        self.assertIsNotNone(parse_agent_match, "Could not find parse_agent_args function")
        
        parse_agent_content = parse_agent_match.group(0)
        # Check that lr_decay_factor is mentioned (either added or checked for)
        self.assertIn('lr_decay_factor', parse_agent_content,
                     "parse_agent_args should handle lr_decay_factor argument")


class TestAgentStructure(unittest.TestCase):
    """Test that agent.py no longer has its own argument parsing."""
    
    def setUp(self):
        """Load the agent.py file content."""
        self.agent_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'src', 
            'DDQN', 
            'agent.py'
        )
        with open(self.agent_path, 'r') as f:
            self.content = f.read()
    
    def test_no_argparse_import(self):
        """Test that agent.py doesn't import argparse."""
        # Check that argparse is not imported at the module level
        lines = self.content.split('\n')
        import_lines = [line for line in lines if line.strip().startswith('import argparse')]
        self.assertEqual(len(import_lines), 0,
                        "agent.py should not import argparse")
    
    def test_no_parser_definition(self):
        """Test that agent.py doesn't define a parser."""
        self.assertNotIn('parser = argparse.ArgumentParser', self.content,
                        "agent.py should not define an ArgumentParser")
    
    def test_no_flags_parsing(self):
        """Test that agent.py doesn't parse FLAGS."""
        self.assertNotIn('FLAGS = parser.parse_args', self.content,
                        "agent.py should not parse FLAGS")
    
    def test_lambda_rule_accepts_parameters(self):
        """Test that lambda_rule function accepts decay_step_size and lr_decay_factor."""
        lambda_match = re.search(
            r'def lambda_rule\([^)]*\)',
            self.content,
            re.MULTILINE
        )
        self.assertIsNotNone(lambda_match, "Could not find lambda_rule function")
        
        lambda_signature = lambda_match.group(0)
        self.assertIn('decay_step_size', lambda_signature,
                     "lambda_rule should accept decay_step_size parameter")
        self.assertIn('lr_decay_factor', lambda_signature,
                     "lambda_rule should accept lr_decay_factor parameter")
    
    def test_agent_init_accepts_flags(self):
        """Test that Agent.__init__ accepts FLAGS parameter."""
        init_match = re.search(
            r'def __init__\([^)]*\)',
            self.content,
            re.MULTILINE | re.DOTALL
        )
        self.assertIsNotNone(init_match, "Could not find Agent.__init__ method")
        
        init_signature = init_match.group(0)
        self.assertIn('FLAGS', init_signature,
                     "Agent.__init__ should accept FLAGS parameter")
        
        # Check that hidden_dim, lr, weight_decay are NOT in signature
        self.assertNotIn('hidden_dim:', init_signature,
                        "Agent.__init__ should not have hidden_dim parameter")
        self.assertNotIn('lr:', init_signature,
                        "Agent.__init__ should not have lr parameter")
        self.assertNotIn('weight_decay:', init_signature,
                        "Agent.__init__ should not have weight_decay parameter")
    
    def test_agent_extracts_all_parameters_from_flags(self):
        """Test that Agent extracts all needed parameters from FLAGS."""
        # Check that hidden_dim, lr, weight_decay are extracted from FLAGS
        self.assertIn('FLAGS.hidden_dim', self.content,
                     "Agent should extract hidden_dim from FLAGS")
        self.assertIn('FLAGS.lr', self.content,
                     "Agent should extract lr from FLAGS")
        self.assertIn('FLAGS.weight_decay', self.content,
                     "Agent should extract weight_decay from FLAGS")
        
        # Check that decay_step_size, lr_decay_factor, and min_lr are extracted
        self.assertIn('self.decay_step_size = FLAGS.decay_step_size', self.content,
                     "Agent should extract decay_step_size from FLAGS")
        self.assertIn('self.lr_decay_factor = FLAGS.lr_decay_factor', self.content,
                     "Agent should extract lr_decay_factor from FLAGS")
        self.assertIn('self.min_lr = FLAGS.min_lr', self.content,
                     "Agent should extract min_lr from FLAGS")
        
        # Check that FLAGS is not stored as a member
        self.assertNotIn('self.FLAGS = FLAGS', self.content,
                        "Agent should not store FLAGS as a member variable")
    
    def test_agent_uses_member_variables(self):
        """Test that Agent uses member variables instead of FLAGS in methods."""
        # Check that update_learning_rate uses self.min_lr
        self.assertIn('self.min_lr', self.content,
                     "Agent should use self.min_lr instead of self.FLAGS.min_lr")
        
        # Check scheduler uses self.decay_step_size and self.lr_decay_factor
        self.assertIn('self.decay_step_size', self.content,
                     "Agent should use self.decay_step_size")
        self.assertIn('self.lr_decay_factor', self.content,
                     "Agent should use self.lr_decay_factor")


class TestMainRobustAgentCall(unittest.TestCase):
    """Test that main_robust.py passes FLAGS to Agent."""
    
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
    
    def test_agent_instantiation_passes_flags(self):
        """Test that Agent is instantiated with FLAGS parameter only."""
        # Find the Agent instantiation
        agent_pattern = r'agent\s*=\s*Agent\([^)]+\)'
        agent_match = re.search(agent_pattern, self.content, re.DOTALL)
        
        self.assertIsNotNone(agent_match, "Could not find Agent instantiation")
        
        agent_call = agent_match.group(0)
        # Should pass only input_dim, output_dim, and FLAGS (3 parameters)
        self.assertIn('FLAGS', agent_call,
                     "Agent should be instantiated with FLAGS parameter")
        
        # Should NOT pass hidden_dim, lr, or weight_decay separately
        self.assertNotIn('FLAGS.hidden_dim', agent_call,
                        "Agent should not receive FLAGS.hidden_dim separately")
        self.assertNotIn('FLAGS.lr', agent_call,
                        "Agent should not receive FLAGS.lr separately")
        self.assertNotIn('FLAGS.weight_decay', agent_call,
                        "Agent should not receive FLAGS.weight_decay separately")


if __name__ == '__main__':
    unittest.main()
