# Test index_functions.py
import unittest
from index_functions import create_metadata_objs

class TestCreateMetadataObjs(unittest.TestCase):

    def test_empty_list(self):
        """Test the function with an empty list."""
        input_list = []
        expected_output = []
        self.assertEqual(create_metadata_objs(input_list), expected_output, "Failed with an empty list")

    def test_non_empty_list(self):
        """Test the function with a non-empty list."""
        input_list = ["Hello", "World"]
        expected_output = [{'text': 'Hello'}, {'text': 'World'}]
        self.assertEqual(create_metadata_objs(input_list), expected_output, "Failed with a non-empty list")

if __name__ == '__main__':
    unittest.main()
