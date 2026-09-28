import json
import unittest

import numpy as np

from llada_study.measurement import summarize_trace


class MeasurementTests(unittest.TestCase):
    def test_six_groups_partition_exactly(self):
        ent = np.array([[.5] * 6, [.6, .4, .6, .4, .5, .5]], dtype=np.float32)
        proposals = np.array([[0] * 6, [1, 1, 0, 0, 1, 0]], dtype=np.int32)
        result = summarize_trace(ent, proposals, np.ones((2, 6), dtype=bool))
        self.assertEqual(result["steps"][1]["counts"], [1] * 6)
        self.assertEqual(result["steps"][1]["common_count"], 6)
        np.testing.assert_allclose(result["features"]["step"], np.ones(6) / 6)
        self.assertFalse(result["steps"][0]["eligible"])
        self.assertEqual(result["steps"][0]["skipped_reason"], "first_step_no_previous")
        self.assertTrue(result["diagnostics"]["partition_counts_valid"])

    def test_survivor_intersection_only(self):
        ent = np.array([[.9, .4, .1, .5], [.1, .6, .9, .5]])
        proposals = np.array([[0, 0, 0, 0], [1, 1, 1, 0]])
        masks = np.array([[1, 1, 0, 0], [0, 1, 1, 0]], dtype=bool)
        row = summarize_trace(ent, proposals, masks)["steps"][1]
        self.assertEqual(row["counts"], [1, 0, 0, 0, 0, 0])
        self.assertEqual(row["common_count"], 1)
        self.assertAlmostEqual(row["mean_entropy_delta"], .2)

    def test_step_and_token_step_averages_differ(self):
        ent = np.array([[.5] * 4, [.6] * 4, [.5, .1, .1, .1]])
        masks = np.array([[1] * 4, [1] * 4, [1, 0, 0, 0]], dtype=bool)
        result = summarize_trace(ent, np.zeros_like(ent, dtype=np.int32), masks)
        np.testing.assert_allclose(result["features"]["step"], [0, 0, .5, .5, 0, 0])
        np.testing.assert_allclose(result["features"]["token_step"], [0, 0, .8, .2, 0, 0])
        self.assertEqual(result["features"]["n_intervals"], 2)
        self.assertEqual(result["features"]["common_count"], 5)

    def test_bins_and_terminal_empty_interval(self):
        masks = np.array([[j < 6-i for j in range(6)] for i in range(7)])
        ent = np.tile(np.arange(7)[:, None] * .05, (1, 6))
        result = summarize_trace(ent, np.zeros((7, 6), dtype=np.int32), masks)
        self.assertEqual([row["region"] for row in result["steps"]],
                         ["many", "many", "middle", "middle", "few", "few", "few"])
        self.assertEqual([result["features"]["regions"][region]["n_intervals"]
                          for region in ("many", "middle", "few")], [1, 2, 2])
        self.assertEqual(result["steps"][-1]["skipped_reason"], "terminal_mask_zero")
        self.assertEqual(result["steps"][-1]["rates"], [0.] * 6)
        self.assertEqual(result["diagnostics"]["n_terminal_mask_zero_intervals"], 1)

    def test_strict_zero_tolerance_tie(self):
        a = np.float32(.5)
        b = np.nextafter(a, np.float32(1.0))
        result = summarize_trace(np.array([[a, a], [b, a]]),
                                 np.zeros((2, 2), dtype=int), np.ones((2, 2), dtype=bool))
        self.assertEqual(result["steps"][1]["counts"], [0, 0, 1, 0, 0, 1])

    def test_undefined_features_are_explicit_json_null(self):
        result = summarize_trace(np.full((2, 2), .5), np.zeros((2, 2), dtype=int),
                                 np.array([[1, 0], [0, 1]], dtype=bool))
        self.assertIsNone(result["features"]["step"])
        self.assertIsNone(result["features"]["token_step"])
        self.assertEqual(result["steps"][1]["skipped_reason"], "empty_survivor_intersection")
        self.assertEqual(result["diagnostics"]["n_skipped_empty_intervals"], 1)
        json.dumps(result, allow_nan=False)

    def test_missing_or_malformed_data_fails_even_on_unmasked_cells(self):
        ent = np.full((2, 2), .5)
        proposals = np.zeros((2, 2), dtype=int)
        masks = np.array([[1, 0], [1, 0]], dtype=bool)
        for bad in (np.nan, np.inf, -np.inf, -1., 2.):
            broken = ent.copy()
            broken[0, 1] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                summarize_trace(broken, proposals, masks)
        with self.assertRaises(ValueError):
            summarize_trace(ent, proposals.astype(float), masks)
        with self.assertRaises(ValueError):
            summarize_trace(ent, proposals, masks.astype(int))
        with self.assertRaises(ValueError):
            summarize_trace(ent, proposals[:, :1], masks)
        with self.assertRaises(ValueError):
            summarize_trace(np.empty((0, 2)), np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=bool))


if __name__ == "__main__":
    unittest.main()
