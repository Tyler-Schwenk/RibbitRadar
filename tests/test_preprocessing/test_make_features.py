import unittest
import torch
import src.inference.prediction_utils as pu
import numpy as np
import wave
import os
from tests.utils import create_test_wav

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        # Generate a short sine wave and save it as a .wav file for testing
        self.test_wav_file = "test_audio.wav"
        create_test_wav(self.test_wav_file)

    def test_make_features(self):
        """
        Test the make_features_fixed function to ensure it returns the expected Mel-frequency features.
        """
        features = pu.make_features_fixed(self.test_wav_file)
        self.assertIsInstance(features, torch.Tensor)
        self.assertEqual(features.shape[1], 128)  # Check for correct Mel bin size
        self.assertTrue(features.shape[0] <= 1000)  # Ensure target length

    def tearDown(self):
        # Remove the test file after running tests
        if os.path.exists(self.test_wav_file):
            os.remove(self.test_wav_file)

if __name__ == '__main__':
    unittest.main()
