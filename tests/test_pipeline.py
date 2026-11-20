import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from predict import SurveyPredictor, prepare_features, risk_level
from threshold_check import recommend, scan_thresholds
from train import BINARY_FEATURES, FEATURES, MODEL_FEATURES, add_symptom_count, load_survey, train

DATA = ROOT / "data" / "lung_cancer.csv"
SAMPLE = ROOT / "data" / "sample_input.csv"


class DataTests(unittest.TestCase):
    def test_load_survey_cleans_real_data(self):
        X, y, counts = load_survey(DATA)
        self.assertEqual(len(X), 276)
        self.assertEqual(counts, {"NO": 38, "YES": 238})
        self.assertEqual(list(X.columns), FEATURES)
        self.assertTrue(X[BINARY_FEATURES].isin([0, 1]).all().all())

    def test_load_survey_rejects_bad_gender(self):
        frame = pd.read_csv(DATA).head(10)
        frame.loc[0, "GENDER"] = "X"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.csv"
            frame.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_survey(path)

    def test_symptom_count_range(self):
        X, _, _ = load_survey(DATA)
        counts = add_symptom_count(X)["SYMPTOM_COUNT"]
        self.assertGreaterEqual(counts.min(), 0)
        self.assertLessEqual(counts.max(), 13)


class PredictInputTests(unittest.TestCase):
    def test_prepare_features_converts_sample(self):
        features = prepare_features(pd.read_csv(SAMPLE))
        self.assertEqual(list(features.columns), MODEL_FEATURES)
        self.assertEqual(features.loc[0, "SYMPTOM_COUNT"], 12)
        self.assertEqual(features.loc[1, "SYMPTOM_COUNT"], 0)

    def test_prepare_features_rejects_bad_age(self):
        frame = pd.read_csv(SAMPLE)
        frame.loc[0, "AGE"] = 150
        with self.assertRaises(ValueError):
            prepare_features(frame)

    def test_risk_level_boundaries(self):
        self.assertEqual(risk_level(0.95), "높음")
        self.assertEqual(risk_level(0.8), "높음")
        self.assertEqual(risk_level(0.6), "주의")
        self.assertEqual(risk_level(0.2), "낮음")


class ThresholdTests(unittest.TestCase):
    def test_recall_never_increases_with_threshold(self):
        y = np.array([0, 0, 1, 1, 1, 1])
        probabilities = np.array([0.1, 0.6, 0.4, 0.7, 0.8, 0.9])
        rows = scan_thresholds(y, probabilities, [0.3, 0.5, 0.75])
        recalls = [row["recall"] for row in rows]
        self.assertEqual(recalls, sorted(recalls, reverse=True))
        self.assertIn(recommend(rows), [0.3, 0.5, 0.75])


class IntegrationTests(unittest.TestCase):
    def test_train_then_predict(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            metrics = train(DATA, output)
            self.assertTrue((output / "metrics.json").exists())
            self.assertTrue((output / "model.joblib").exists())
            saved = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["test_rows"], 56)
            self.assertGreaterEqual(metrics["accuracy"], 0.85)
            self.assertGreaterEqual(metrics["roc_auc"], 0.9)

            predictor = SurveyPredictor(output / "model.joblib")
            table = predictor.predict(pd.read_csv(SAMPLE))
            self.assertEqual(len(table), 5)
            self.assertTrue(table["PREDICTION"].isin(["YES", "NO"]).all())
            self.assertTrue(table["YES_PROBABILITY"].between(0, 1).all())

    def test_missing_model_gives_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            SurveyPredictor(Path("no_such_model.joblib"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
