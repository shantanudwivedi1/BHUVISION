import io
import unittest
from PIL import Image

from backend.router.grounding_dino import (
    DEFAULT_REMOTE_SENSING_CLASSES,
    GroundingDINOAdapter,
    GroundingDINOConfig,
    _prepare_grounding_text,
    _render_bounding_boxes,
)


class TestGroundingDINO(unittest.TestCase):
    def test_config_defaults(self):
        cfg = GroundingDINOConfig()
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.model_id, "IDEA-Research/grounding-dino-tiny")
        self.assertEqual(cfg.box_threshold, 0.25)
        self.assertEqual(cfg.text_threshold, 0.25)

    def test_prepare_grounding_text_from_query(self):
        # Query with detect phrase
        t1 = _prepare_grounding_text("Detect all the buildings and aircraft in this satellite image.")
        self.assertEqual(t1, "buildings. aircraft.")

        # Query with highlight
        t2 = _prepare_grounding_text("Highlight roads, vehicles, and runways with bounding boxes.")
        self.assertEqual(t2, "roads. vehicles. runways.")

        # Empty query falls back to default remote sensing classes
        t3 = _prepare_grounding_text("")
        self.assertTrue(t3.endswith("."))
        for c in DEFAULT_REMOTE_SENSING_CLASSES:
            self.assertIn(c, t3)

    def test_adapter_descriptor(self):
        adapter = GroundingDINOAdapter()
        desc = adapter.describe()
        self.assertEqual(desc.model_name, "Grounding DINO")
        self.assertIn("object_detection", desc.capabilities)
        self.assertIn("open_vocabulary", desc.capabilities)
        self.assertIn("optical", desc.required_modalities)
        self.assertTrue(desc.available)

    def test_render_bounding_boxes(self):
        img = Image.new("RGB", (200, 200), color=(100, 150, 200))
        detections = [
            {
                "class": "building",
                "confidence": 0.88,
                "bbox": [20, 20, 80, 80],
                "box_pct": [10.0, 10.0, 40.0, 40.0],
            }
        ]
        data_url = _render_bounding_boxes(img, detections)
        self.assertTrue(data_url.startswith("data:image/png;base64,"))
        self.assertGreater(len(data_url), 100)


if __name__ == "__main__":
    unittest.main()
