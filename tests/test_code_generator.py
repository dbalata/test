import unittest
from unittest.mock import patch, MagicMock

from src.code_generator.core import CodeGenerator, CodeGenerationError
from src.code_generator.models.generation import GenerationResult, CodeBlock
from src.code_generator.testing import generate_testing_suite

class TestCodeGeneratorTesting(unittest.TestCase):

    @patch('src.code_generator.testing.CodeGenerator')
    def test_generate_python_test_suite(self, MockCodeGenerator):
        mock_instance = MockCodeGenerator.return_value
        expected_test_code = "def test_example():\n    assert True"
        mock_generation_result = GenerationResult(
            explanation="Python tests",
            code_blocks=[CodeBlock(language="python", code=expected_test_code)],
            dependencies=[],
            usage_examples=[]
        )
        mock_instance.generate_testing_suite.return_value = mock_generation_result

        sample_python_code = "def hello():\n    return 'world'"
        result = generate_testing_suite(code=sample_python_code, language="python")

        mock_instance.generate_testing_suite.assert_called_once_with(
            code_to_test=sample_python_code,
            testing_framework="pytest"
        )
        self.assertEqual(result, expected_test_code)

    @patch('src.code_generator.testing.CodeGenerator')
    def test_generate_javascript_test_suite(self, MockCodeGenerator):
        mock_instance = MockCodeGenerator.return_value
        expected_test_code = "test('example', () => { expect(true).toBe(true); });"
        mock_generation_result = GenerationResult(
            explanation="JavaScript tests",
            code_blocks=[CodeBlock(language="javascript", code=expected_test_code)],
            dependencies=[],
            usage_examples=[]
        )
        mock_instance.generate_testing_suite.return_value = mock_generation_result

        sample_js_code = "function greet() { return 'hello'; }"
        result = generate_testing_suite(code=sample_js_code, language="javascript")

        mock_instance.generate_testing_suite.assert_called_once_with(
            code_to_test=sample_js_code,
            testing_framework="jest"
        )
        self.assertEqual(result, expected_test_code)
        
    @patch('src.code_generator.testing.CodeGenerator')
    def test_generate_other_language_test_suite_defaults_to_pytest(self, MockCodeGenerator):
        mock_instance = MockCodeGenerator.return_value
        expected_test_code = "def test_example_other_lang():\n    assert True"
        mock_generation_result = GenerationResult(
            explanation="Other language tests with pytest",
            code_blocks=[CodeBlock(language="python", code=expected_test_code)],
            dependencies=[],
            usage_examples=[]
        )
        mock_instance.generate_testing_suite.return_value = mock_generation_result

        sample_ruby_code = "def foo\n  'bar'\nend"
        result = generate_testing_suite(code=sample_ruby_code, language="ruby") # language not explicitly mapped

        mock_instance.generate_testing_suite.assert_called_once_with(
            code_to_test=sample_ruby_code,
            testing_framework="pytest" # Should default to pytest
        )
        self.assertEqual(result, expected_test_code)


    @patch('src.code_generator.testing.CodeGenerator')
    def test_generate_test_suite_no_code_block(self, MockCodeGenerator):
        mock_instance = MockCodeGenerator.return_value
        mock_generation_result = GenerationResult(
            explanation="No code blocks",
            code_blocks=[] # Empty list
        )
        mock_instance.generate_testing_suite.return_value = mock_generation_result

        sample_code = "let x = 10;"
        result = generate_testing_suite(code=sample_code, language="javascript")

        mock_instance.generate_testing_suite.assert_called_once_with(
            code_to_test=sample_code,
            testing_framework="jest"
        )
        self.assertEqual(result, "# No test suite code was generated.")

    @patch('src.code_generator.testing.CodeGenerator')
    def test_generate_test_suite_llm_error(self, MockCodeGenerator):
        mock_instance = MockCodeGenerator.return_value
        error_message = "LLM failed spectacularly"
        # We use a generic Exception here as CodeGenerationError is from .core and might cause import issues
        # or be too specific if the actual method in core.py raises a different subclass of Exception.
        # The function in testing.py catches `Exception as e`.
        mock_instance.generate_testing_suite.side_effect = Exception(error_message)

        sample_code = "class MyClass {}"
        result = generate_testing_suite(code=sample_code, language="python")

        mock_instance.generate_testing_suite.assert_called_once_with(
            code_to_test=sample_code,
            testing_framework="pytest"
        )
        self.assertEqual(result, f"# Error generating test suite: {error_message}")

if __name__ == '__main__':
    unittest.main()
