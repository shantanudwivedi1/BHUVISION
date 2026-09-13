import unittest

from backend.confidence.report import build_confidence_report
from backend.evidence.aggregation import aggregate_evidence
from backend.models.schemas import Evidence


class EvidenceConfidenceTests(unittest.TestCase):
    def test_evidence_preserves_provenance_fields(self):
        item = Evidence(
            evidence_type="model_finding",
            source="scene.tif",
            model="test-model",
            finding="water",
            spatial={"bbox": [0, 0, 10, 10], "coordinate_system": "pixel"},
            confidence=0.8,
            provenance={"run_id": "r1"},
        )
        aggregate = aggregate_evidence([item])
        preserved = aggregate.evidence[0]
        self.assertEqual(preserved.model, "test-model")
        self.assertEqual(preserved.finding, "water")
        self.assertEqual(preserved.spatial["bbox"], [0, 0, 10, 10])
        self.assertEqual(preserved.provenance["run_id"], "r1")
        self.assertEqual(aggregate.findings, ["water"])

    def test_missing_confidence_signals_are_explicit(self):
        report = build_confidence_report(
            [Evidence(evidence_type="image_metadata", source="scene.png")],
            image_analysis=[
                {
                    "format": "PNG",
                    "width": 640,
                    "height": 480,
                    "quality": {"file_size_bytes": 100, "warnings": []},
                }
            ],
        )
        signals = {signal.name: signal for signal in report.signals}
        self.assertEqual(signals["model_confidence"].status, "unavailable")
        self.assertEqual(signals["model_agreement"].status, "unavailable")
        self.assertEqual(signals["spatial_consistency"].status, "unavailable")
        self.assertEqual(signals["paired_image_alignment"].status, "unavailable")
        self.assertEqual(signals["input_image_quality"].status, "available")
        self.assertIsNone(signals["input_image_quality"].value)
        self.assertIsNone(report.aggregate_value)

    def test_model_confidence_agreement_and_bbox_overlap_are_deterministic(self):
        evidence = [
            Evidence(
                evidence_type="model_finding",
                source="scene.tif",
                model="model-a",
                finding="water",
                confidence=0.8,
                spatial={"bbox": [0, 0, 10, 10]},
            ),
            Evidence(
                evidence_type="model_finding",
                source="scene.tif",
                model="model-b",
                finding="water",
                confidence=0.6,
                spatial={"bbox": [5, 0, 15, 10]},
            ),
        ]
        report = build_confidence_report(evidence)
        signals = {signal.name: signal for signal in report.signals}
        self.assertEqual(signals["model_confidence"].value, 0.7)
        self.assertEqual(signals["model_agreement"].value, 1.0)
        self.assertEqual(signals["spatial_consistency"].value, round(1 / 3, 6))
        self.assertEqual(report.aggregate_value, round((0.7 + 1.0 + (1 / 3)) / 3, 6))


if __name__ == "__main__":
    unittest.main()
