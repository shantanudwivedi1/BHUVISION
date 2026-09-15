"""Unit tests for GeoChat adapter, output parsing, bounding box rendering, and routing."""

import io
import unittest
from PIL import Image

from backend.models.schemas import QueryPlan
from backend.router.geochat import (
    GeoChatAdapter,
    GeoChatConfig,
    _parse_geochat_output,
    _render_bounding_boxes,
)
from backend.router.model_router import _split_optical_and_sar, route_plan


class GeoChatTests(unittest.TestCase):
    def test_geochat_describe(self):
        adapter = GeoChatAdapter()
        desc = adapter.describe().as_dict()
        self.assertEqual(desc["model_name"], "GeoChat-7B")
        self.assertIn("vqa", desc["capabilities"])
        self.assertIn("captioning", desc["capabilities"])
        self.assertIn("object_detection", desc["capabilities"])
        self.assertIn("optical", desc["required_modalities"])
        self.assertIn("sar", desc["required_modalities"])
        self.assertTrue(desc["available"])

    def test_geochat_disabled_describe(self):
        adapter = GeoChatAdapter(GeoChatConfig(enabled=False))
        desc = adapter.describe()
        self.assertFalse(desc.available)
        self.assertIn("disabled", desc.reason.lower())

    def test_parse_geochat_output_grounded_boxes(self):
        raw = "In the satellite image, there are <p>buildings</p> {<10><20><50><60>|<90>}<delim>{<12><22><48><58>|<90>} and <p>runway</p> {<70><10><90><95>|<0>}."
        clean_text, detections = _parse_geochat_output(raw)
        self.assertEqual(clean_text, "In the satellite image, there are buildings and runway.")
        self.assertEqual(len(detections), 3)
        self.assertEqual(detections[0]["class"], "buildings")
        self.assertEqual(detections[0]["box_pct"], [10, 20, 50, 60])
        self.assertEqual(detections[2]["class"], "runway")
        self.assertEqual(detections[2]["box_pct"], [70, 10, 90, 95])

    def test_render_bounding_boxes(self):
        img = Image.new("RGB", (200, 100), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        detections = [
            {"class": "aircraft", "confidence": 0.95, "box_pct": [20, 10, 80, 50]}
        ]
        data_url = _render_bounding_boxes(image_bytes, detections)
        self.assertTrue(data_url.startswith("data:image/png;base64,"))
        self.assertEqual(detections[0]["bbox"], [20, 20, 100, 80])

    def test_split_optical_and_sar(self):
        uploads = [
            ("optical_scene.png", "image/png", b"optical_data"),
            ("sentinel1_sar_vh.png", "image/png", b"sar_data"),
        ]
        optical, sar = _split_optical_and_sar(uploads)
        self.assertEqual(len(optical), 1)
        self.assertEqual(optical[0][0], "optical_scene.png")
        self.assertEqual(sar, b"sar_data")

    def test_route_plan_prefers_geochat(self):
        plan_vqa = QueryPlan(task="vqa", intent="What features are present?", modality_requirements=["optical"])
        route_vqa = route_plan(plan_vqa)
        self.assertEqual(route_vqa["requested_capability"], "GeoChat-7B")
        self.assertTrue(route_vqa["available"])

        plan_cap = QueryPlan(task="captioning", intent="Describe the scene.", modality_requirements=["optical"])
        route_cap = route_plan(plan_cap)
        self.assertEqual(route_cap["requested_capability"], "GeoChat-7B")
        self.assertTrue(route_cap["available"])

        plan_det = QueryPlan(task="object_detection", intent="Detect storage tanks.", modality_requirements=["optical"])
        route_det = route_plan(plan_det)
        self.assertIn("GeoChat-7B", route_det["requested_capability"])
        self.assertIn("YOLO11", route_det["requested_capability"])


if __name__ == "__main__":
    unittest.main()
