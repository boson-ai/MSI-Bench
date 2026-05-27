import json
import tempfile
import unittest
from pathlib import Path

from scripts.score_predictions import normalize_answer, safe_div
from scripts.validate_manifest import candidate_roots, resolve_existing_path


class ValidateManifestTests(unittest.TestCase):
    def test_candidate_roots_include_manifest_ancestors_for_workspace_relative_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            repo = workspace / "Interaction-Bench"
            examples = repo / "examples"
            examples.mkdir(parents=True)
            audio = workspace / "HumDial-FDBench" / "test" / "sample.wav"
            audio.parent.mkdir(parents=True)
            audio.write_bytes(b"RIFF")
            manifest = examples / "seed_manifest.jsonl"
            manifest.write_text("", encoding="utf-8")

            roots = candidate_roots(manifest, None)
            self.assertEqual(
                resolve_existing_path("HumDial-FDBench/test/sample.wav", roots),
                audio,
            )


class ScorePredictionTests(unittest.TestCase):
    def test_normalize_answer_removes_articles_punctuation_and_case(self):
        self.assertEqual(normalize_answer("The Quick, brown fox!"), "quick brown fox")

    def test_safe_div_handles_empty_denominator(self):
        self.assertIsNone(safe_div(1, 0))
        self.assertEqual(safe_div(3, 2), 1.5)


if __name__ == "__main__":
    unittest.main()
