# tests/test_app/test_model_manager.py
import unittest
import os
import shutil
from src.app.model_manager import update_local_model, get_latest_local_model_version

class TestModelManager(unittest.TestCase):
    def setUp(self):
        # Setup a temporary model directory for testing
        self.model_dir = "test_model_dir"
        os.makedirs(self.model_dir, exist_ok=True)

        # Create a model file with the correct naming pattern
        self.test_model_file = os.path.join(self.model_dir, "best_audio_model_V7.pth")
        with open(self.test_model_file, "w") as f:
            f.write("dummy model content")

    def tearDown(self):
        # Clean up test model directory after test
        shutil.rmtree(self.model_dir)

    def test_update_local_model(self):
        """
        Test that the local model is updated correctly.
        """
        update_local_model(self.model_dir, lambda msg: None)
        self.assertTrue(os.path.exists(self.test_model_file))

    def test_get_latest_local_model_version(self):
        """
        Test that the latest local model version is correctly identified.
        """
        latest_version = get_latest_local_model_version(self.model_dir)
        self.assertEqual(latest_version, 7)

if __name__ == '__main__':
    unittest.main()
