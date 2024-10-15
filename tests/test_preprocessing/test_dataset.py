# tests/test_preprocessing/test_dataset.py
import unittest
import os
from src.preprocessing.dataset import RanaDraytoniiDataset, get_data_loader

class TestDataset(unittest.TestCase):
    def setUp(self):
        # Create a dummy directory and files for testing
        self.test_dir = "test_audio_dir"
        os.makedirs(self.test_dir, exist_ok=True)
        self.test_audio_file = os.path.join(self.test_dir, "test_audio.wav")
        with open(self.test_audio_file, "w") as f:
            f.write("dummy audio content")

    def tearDown(self):
        # Clean up test files after test
        if os.path.exists(self.test_audio_file):
            os.remove(self.test_audio_file)
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)

    def test_dataset_initialization(self):
        """
        Test the initialization of the RanaDraytoniiDataset.
        """
        dataset = RanaDraytoniiDataset(self.test_dir)
        self.assertEqual(len(dataset), 1)
        self.assertTrue(self.test_audio_file in dataset.files)

    def test_get_data_loader(self):
        """
        Test that the data loader works and loads batches correctly.
        """
        dataset = RanaDraytoniiDataset(self.test_dir)
        data_loader = get_data_loader(self.test_dir, batch_size=1, shuffle=False)
        for batch in data_loader:
            self.assertEqual(len(batch), 1)

if __name__ == '__main__':
    unittest.main()
