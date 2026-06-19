import math
import unittest

import awkward as ak
import numpy as np
import pandas as pd

from heavy_ion_alice_analysis import (
    centrality_to_numpy,
    compare_dndeta_to_reference,
    compute_dn_deta,
    event_multiplicity,
    histogram_density,
    sampled_pair_deltas,
    validation_report,
    v2_two_particle,
)


class AnalysisHelperTests(unittest.TestCase):
    def test_event_multiplicity_counts_tracks_per_event(self):
        tracks = ak.Array(
            [
                {"pt": [0.2, 0.4], "eta": [0.1, -0.2]},
                {"pt": [0.8], "eta": [0.3]},
                {"pt": [], "eta": []},
            ]
        )

        np.testing.assert_array_equal(event_multiplicity(tracks), np.array([2, 1, 0]))

    def test_centrality_to_numpy_accepts_jagged_event_values(self):
        cent = ak.Array([[2.5], [17.0], [52.0]])

        np.testing.assert_allclose(
            centrality_to_numpy(cent, n_events=3),
            np.array([2.5, 17.0, 52.0]),
        )

    def test_histogram_density_normalizes_by_events_and_bin_width(self):
        values = np.array([-0.5, 0.2, 0.4])
        edges = np.array([-1.0, 0.0, 1.0])

        table = histogram_density(values, edges, n_events=2)

        np.testing.assert_array_equal(table["counts"].to_numpy(), np.array([1, 2]))
        np.testing.assert_allclose(
            table["per_event_density"].to_numpy(),
            np.array([0.5, 1.0]),
        )

    def test_compute_dn_deta_labels_output(self):
        tracks = ak.Array(
            [
                {"pt": [0.2, 0.4], "eta": [-0.5, 0.5]},
                {"pt": [0.8], "eta": [0.5]},
            ]
        )

        table = compute_dn_deta(tracks, np.array([-1.0, 0.0, 1.0]), label="0-5%")

        self.assertEqual(table["centrality"].tolist(), ["0-5%", "0-5%"])
        np.testing.assert_array_equal(table["counts"].to_numpy(), np.array([1, 2]))

    def test_v2_two_particle_uses_q_vector_identity(self):
        tracks = ak.Array(
            [
                {
                    "pt": [0.2, 0.4],
                    "eta": [0.1, -0.1],
                    "phi": [0.0, math.pi],
                }
            ]
        )

        result = v2_two_particle(tracks)

        self.assertEqual(result["events_used"], 1)
        self.assertEqual(result["pairs_used"], 2)
        self.assertAlmostEqual(result["c2"], 1.0)
        self.assertAlmostEqual(result["v2_2"], 1.0)

    def test_sampled_pair_deltas_wraps_delta_phi_to_minus_pi_pi(self):
        tracks = ak.Array(
            [
                {
                    "pt": [0.2, 0.4],
                    "eta": [0.6, -0.4],
                    "phi": [math.pi - 0.1, -math.pi + 0.1],
                }
            ]
        )

        delta_eta, delta_phi = sampled_pair_deltas(
            tracks,
            max_events=10,
            max_tracks=10,
            seed=7,
        )

        np.testing.assert_allclose(delta_eta, np.array([1.0]))
        np.testing.assert_allclose(delta_phi, np.array([-0.2]))

    def test_validation_report_flags_missing_reference_comparison(self):
        tracks = ak.Array(
            [
                {
                    "pt": [0.2, 0.4],
                    "eta": [0.1, -0.1],
                    "phi": [0.0, math.pi],
                }
            ]
        )
        centrality_df = pd.DataFrame(
            [
                {
                    "centrality": "all",
                    "v2_events_used": 1,
                    "mean_multiplicity": 2.0,
                }
            ]
        )
        summary_df = pd.DataFrame([{"phi_present": True}])

        report = validation_report(tracks, None, centrality_df, summary_df)

        reference = report[report["check"] == "reference_comparison"].iloc[0]
        self.assertEqual(reference["status"], "fail")

    def test_compare_dndeta_to_reference_passes_matching_value(self):
        dn_deta = pd.DataFrame(
            [
                {
                    "centrality": "0-5%",
                    "bin_center": -0.25,
                    "dN_deta": 1940.0,
                },
                {
                    "centrality": "0-5%",
                    "bin_center": 0.25,
                    "dN_deta": 1946.0,
                },
            ]
        )
        reference = pd.DataFrame(
            [
                {
                    "observable": "dN_deta_midrapidity",
                    "centrality": "0-5%",
                    "centrality_low": 0,
                    "centrality_high": 5,
                    "eta_abs_max": 0.5,
                    "value": 1943.0,
                    "uncertainty_total": 56.0,
                    "tolerance_relative": 0.30,
                    "tolerance_sigma": 3.0,
                    "source": "test",
                    "derivation": "test",
                }
            ]
        )
        path = self._reference_csv(reference)

        comparison = compare_dndeta_to_reference(dn_deta, reference_path=path)

        self.assertEqual(comparison.iloc[0]["status"], "pass")
        self.assertAlmostEqual(comparison.iloc[0]["measured_value"], 1943.0)

    def _reference_csv(self, frame):
        import tempfile

        handle = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="")
        try:
            frame.to_csv(handle.name, index=False)
            return handle.name
        finally:
            handle.close()


if __name__ == "__main__":
    unittest.main()
