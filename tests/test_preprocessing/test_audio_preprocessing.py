# tests/test_preprocessing/test_audio_preprocessing.py
import unittest
import os
import shutil
from src.preprocessing.audio_preprocessing import (
    resampler,
    split_all_audio_files,
    clear_directory
)
from tests.utils import create_test_wav

class TestAudioPreprocessing(unittest.TestCase):
    def setUp(self):
        # Setup a dummy directory and files for testing
        self.test_dir = "test_audio_dir"
        os.makedirs(self.test_dir, exist_ok=True)
        self.test_audio_file = os.path.join(self.test_dir, "test_audio.wav")
        create_test_wav(self.test_audio_file)

        self.output_dir = "output_audio_dir"
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        # Clean up test files and directories after test
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_resampler(self):
        """
        Test that the audio resampling process works.
        """
        resampler(self.test_audio_file, self.output_dir)
        output_files = os.listdir(self.output_dir)
        self.assertEqual(len(output_files), 1)
        self.assertTrue(output_files[0].endswith(".wav"))

    def dummy_progress_callback(self, *args):
        pass

    def test_split_all_audio_files(self):
        """
        Test that audio is split into segments properly.
        """
        split_all_audio_files(self.test_dir, self.output_dir, self.dummy_progress_callback, segment_length_sec=10)
        output_files = os.listdir(self.output_dir)
        self.assertGreater(len(output_files), 0)

    def test_clear_directory(self):
        """
        Test that the clear_directory function removes all files in a directory.
        """
        clear_directory(self.output_dir)
        self.assertEqual(len(os.listdir(self.output_dir)), 0)

if __name__ == '__main__':
    unittest.main()
