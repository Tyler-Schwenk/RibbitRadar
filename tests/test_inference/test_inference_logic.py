# tests/test_inference/test_inference_logic.py
import unittest
from src.inference.prediction_utils import determine_prediction

class TestInferenceLogic(unittest.TestCase):
    def test_determine_prediction_highest_score(self):
        """
        Test the 'Highest Score' prediction mode.
        """
        label_choice = ["RADR", "RACA", "Negative"]
        scores = [0.7, 0.3, 0.1]
        result = determine_prediction(scores, radr_threshold=0.5, raca_threshold=0.5, prediction_mode="Highest Score", label_choice=label_choice)
        self.assertEqual(result, "RADR")

    def test_determine_prediction_threshold(self):
        """
        Test the 'Threshold' prediction mode.
        """
        label_choice = ["RADR", "RACA", "Negative"]
        scores = [0.6, 0.7, 0.1]
        result = determine_prediction(scores, radr_threshold=0.5, raca_threshold=0.5, prediction_mode="Threshold", label_choice=label_choice)
        self.assertEqual(result, "RADR, RACA")

    def test_determine_prediction_negative(self):
        """
        Test that 'Negative' is predicted if thresholds are not met.
        """
        label_choice = ["RADR", "RACA", "Negative"]
        scores = [0.4, 0.3, 0.8]
        result = determine_prediction(scores, radr_threshold=0.5, raca_threshold=0.5, prediction_mode="Threshold", label_choice=label_choice)
        self.assertEqual(result, "Negative")

if __name__ == '__main__':
    unittest.main()
