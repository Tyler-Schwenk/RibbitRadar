# tests/test_output/test_save_results.py
import unittest
import os
import pandas as pd
from src.inference.result_processing import save_results

class TestSaveResults(unittest.TestCase):
    def setUp(self):
        # Setup paths for test
        self.output_file = "test_results.xlsx"
        self.test_results = pd.DataFrame({
            'File Name': ['test_file'],
            'Prediction': ['RADR'],
            'Times Heard RACA': ['N/A'],
            'Times Heard RADR': ['N/A'],
            'Device ID': ['TestDevice'],
            'Timestamp': ['2024-10-11'],
            'Temperature': [25.0],
            'Segment': ['N/A'],
        })

    def tearDown(self):
        # Clean up the output file after test
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def test_save_results(self):
        """
        Test that results are saved correctly to an Excel file.
        """
        save_results(
            results_df=self.test_results,
            results_path=self.output_file,
            model_version="v1.0",
            raca_threshold=0.5,
            radr_threshold=0.5,
            full_report=True,
            summary_report=False,
            custom_report={},
            label_choice=["RADR", "RACA", "Negative"],
            prediction_mode="Highest Score"
        )
        # Check that file is created
        self.assertTrue(os.path.exists(self.output_file))

if __name__ == '__main__':
    unittest.main()
