import unittest

from backend.evidence.image_analysis import inspect_image
from pydantic import ValidationError

from backend.planner.llm import LLMPlannerConfig, LLMPlannerError, LLMQueryPlanner, LLMTaskResponse
from backend.planner.planner import build_query_plan, build_query_plan_with_optional_llm
from backend.router.model_router import route_plan


class PlannerTests(unittest.TestCase):
    def test_required_query_patterns(self):
        cases = {
            "What is visible in this satellite image?": "vqa",
            "Describe this image.": "captioning",
            "What new buildings appeared between these two images?": "change_detection",
            "Compare these two images and show what changed.": "change_detection",
            "Detect all objects in this image.": "object_detection",
            "Highlight the buildings in this scene.": "object_detection",
            "Find vehicles in this image.": "object_detection",
            "Show the ships in this image.": "vqa",
            "Hello, how are you?": "unsupported",
        }
        for query, task in cases.items():
            with self.subTest(query=query):
                self.assertEqual(build_query_plan(query).task, task)

    def test_fields_are_inferred_without_inventing_modalities(self):
        visible = build_query_plan("What is visible in this satellite image?")
        self.assertEqual(visible.image_count, 1)
        self.assertFalse(visible.temporal)
        self.assertEqual(visible.modalities, [])
        self.assertEqual(visible.output_types, ["answer", "evidence"])

        change = build_query_plan("What new buildings appeared between these two images?")
        self.assertEqual(change.objects, ["buildings"])
        self.assertEqual(change.image_count, 2)
        self.assertTrue(change.temporal)
        self.assertEqual(change.modalities, [])

        ships = build_query_plan("Show the ships in this image.")
        self.assertEqual(ships.objects, ["ships"])
        self.assertEqual(ships.output_types, ["object_list", "evidence"])

    def test_highlight_and_detect_trigger_object_detection(self):
        """Queries with highlight/detect/mark + objects should route to YOLO."""
        cases = {
            "Highlight buildings in this image.": "object_detection",
            "Detect cars in this scene.": "object_detection",
            "Mark all the vehicles.": "object_detection",
            "Spot the ships in this image.": "object_detection",
            "Outline the roads.": "object_detection",
            "Locate the buildings.": "object_detection",
        }
        for query, task in cases.items():
            with self.subTest(query=query):
                self.assertEqual(build_query_plan(query).task, task)

    def test_query_plan_validation_controls_values(self):
        with self.assertRaises(ValidationError):
            build_query_plan("Describe this image.", modalities=["made_up_sensor"])

    def test_numeric_image_count_inference(self):
        self.assertEqual(build_query_plan("Compare these three images.").image_count, 3)

    def test_optional_llm_planner_handles_existing_queries_without_model_selection(self):
        cases = {
            "What is visible in this satellite image?": "vqa",
            "Describe this image.": "captioning",
            "What new buildings appeared between these two images?": "change_detection",
            "Compare these two images and show what changed.": "change_detection",
            "Detect all objects in this image.": "object_detection",
            "Show the ships in this image.": "vqa",
            "Hello, how are you?": "unsupported",
        }

        class FakeLLM:
            def plan_task(self, query):
                return cases[query]

        for query, expected_task in cases.items():
            with self.subTest(query=query):
                plan = build_query_plan_with_optional_llm(query, llm_planner=FakeLLM())
                self.assertEqual(plan.task, expected_task)

    def test_optional_llm_failure_falls_back_to_deterministic_plan(self):
        class FailingLLM:
            def plan_task(self, _query):
                raise LLMPlannerError("test failure")

        deterministic = build_query_plan("Describe this image.")
        fallback = build_query_plan_with_optional_llm("Describe this image.", llm_planner=FailingLLM())
        self.assertEqual(fallback, deterministic)

    def test_llm_task_output_is_schema_validated(self):
        self.assertEqual(LLMTaskResponse(task="vqa").task, "vqa")
        with self.assertRaises(ValidationError):
            LLMTaskResponse(task="not_a_task")
        with self.assertRaises(ValidationError):
            LLMTaskResponse(task="vqa", modality="optical")

    def test_llm_is_disabled_by_default(self):
        planner = LLMQueryPlanner(LLMPlannerConfig())
        with self.assertRaises(LLMPlannerError):
            planner.plan_task("Describe this image.")

    def test_model_router_marks_semantic_capabilities_unavailable(self):
        plan = build_query_plan("What is visible in this satellite image?")
        route = route_plan(plan)
        # VQA routes to GeoChat-7B (sole VLM, no BLIP fallback).
        self.assertEqual(route["requested_capability"], "GeoChat-7B")
        self.assertTrue(route["available"])
        self.assertEqual(route["execution"], "not_executed")

    def test_object_detection_routes_to_yolo(self):
        plan = build_query_plan("Detect all objects in this image.")
        route = route_plan(plan)
        self.assertIn("YOLO11", route["requested_capability"])
        self.assertTrue(route["available"])

    def test_png_metadata_is_real_and_nonsemantic(self):
        png_header = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR" + (640).to_bytes(4, "big") + (480).to_bytes(4, "big") + b"\x08\x02"
        info = inspect_image(png_header, "scene.png")
        self.assertEqual(info["format"], "PNG")
        self.assertEqual((info["width"], info["height"], info["channels"]), (640, 480, 3))
        self.assertIn("file_size_bytes", info["quality"])


if __name__ == "__main__":
    unittest.main()
