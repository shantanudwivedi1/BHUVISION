import unittest

from backend.router.croma import CROMAConfig, CROMAAdapter, CROMAPreprocessingError, prepare_croma_inputs
from backend.models.schemas import QueryPlan
from backend.router.model_router import route_plan


def _cube(channels, height=2, width=3, value=0):
    return [[[value for _ in range(width)] for _ in range(height)] for _ in range(channels)]


class CROMATests(unittest.TestCase):
    def test_preprocessing_accepts_expected_channels(self):
        prepared = prepare_croma_inputs(_cube(2), _cube(12))
        self.assertEqual(prepared.sar_shape, (2, 2, 3))
        self.assertEqual(prepared.optical_shape, (12, 2, 3))

    def test_preprocessing_removes_b10_from_13_band_optical_input(self):
        optical = _cube(13)
        optical[10][0][0] = 999
        prepared = prepare_croma_inputs(_cube(2), optical)
        self.assertEqual(prepared.optical_shape, (12, 2, 3))
        self.assertNotEqual(prepared.optical_images[10][0][0], 999)

    def test_preprocessing_rejects_wrong_channels_or_spatial_size(self):
        with self.assertRaises(CROMAPreprocessingError):
            prepare_croma_inputs(_cube(1), _cube(12))
        with self.assertRaises(CROMAPreprocessingError):
            prepare_croma_inputs(_cube(2), _cube(11))
        with self.assertRaises(CROMAPreprocessingError):
            prepare_croma_inputs(_cube(2, 2, 3), _cube(12, 2, 4))

    def test_default_adapter_is_honestly_unavailable(self):
        result = CROMAAdapter(CROMAConfig()).run(_cube(2), _cube(12))
        self.assertEqual(result.status, "capability_unavailable")
        self.assertIn("disabled", result.explanation.lower())

    def test_router_exposes_croma_contract(self):
        routing = route_plan(QueryPlan(task="optical_sar_analysis", modalities=["optical", "sar"], image_count=2))
        self.assertEqual(routing["model_name"], "CROMA")
        self.assertEqual(routing["required_modalities"], ["optical", "sar"])
        self.assertEqual(routing["output_type"], "learned_representation/features")
        self.assertFalse(routing["available"])


if __name__ == "__main__":
    unittest.main()
