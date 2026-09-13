import unittest

from pydantic import ValidationError

from backend.models.schemas import AnalysisRequest, QueryPlan


class ModelTests(unittest.TestCase):
    def test_query_plan_shape(self):
        plan = QueryPlan(task="vqa", image_count=1, modalities=["optical"])
        self.assertEqual(plan.task, "vqa")
        self.assertEqual(plan.objects, [])

    def test_analysis_request_requires_query(self):
        with self.assertRaises(ValueError):
            AnalysisRequest(query="")

    def test_query_plan_rejects_uncontrolled_values(self):
        with self.assertRaises(ValidationError):
            QueryPlan(task="made_up_task")
        with self.assertRaises(ValidationError):
            QueryPlan(task="vqa", modalities=["made_up_modality"])
        with self.assertRaises(ValidationError):
            QueryPlan(task="vqa", output_types=["made_up_output"])


if __name__ == "__main__":
    unittest.main()
