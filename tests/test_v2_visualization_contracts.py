from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = PROJECT_ROOT / "src" / "08_multi_objective"
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from helper.visualization import generate_proposal_artifacts  # noqa: E402


PROTOTYPE_SCRIPT = (
    V2_ROOT / "04_report_campaign" / "prototype_visualizations.py"
)
PROPOSAL = (
    PROJECT_ROOT
    / "results"
    / "multi_objective_v2"
    / "next_round"
    / "next_round_candidates.csv"
)
CONCEPT_DIRS = [
    "concept_A_feasible_pareto",
    "concept_B_campaign_timeline",
    "concept_C_surrogate_trust",
    "concept_D_candidate_decision",
    "concept_E_publication_triptych",
]


class V2VisualizationContractTests(unittest.TestCase):
    def test_compatibility_proposal_name_and_dimensions_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            generated = generate_proposal_artifacts(
                pd.read_csv(PROPOSAL),
                Path(temporary_name),
            )
            self.assertEqual(
                [path.name for path in generated],
                ["next_round_candidate_screen.png"],
            )
            with Image.open(generated[0]) as image:
                self.assertEqual(image.size, (1440, 1080))

    def test_gallery_outputs_and_scientific_display_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            output = Path(temporary_name) / "visualization_concepts"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROTOTYPE_SCRIPT),
                    "--output-dir",
                    str(output),
                    "--seed",
                    "42",
                ],
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((output / "concept_comparison.png").exists())
            for directory_name in CONCEPT_DIRS:
                directory = output / directory_name
                pngs = list(directory.glob("*.png"))
                pdfs = list(directory.glob("*.pdf"))
                tables = list(directory.glob("*_tidy_data.csv"))
                self.assertEqual(len(pngs), 1, directory_name)
                self.assertEqual(len(pdfs), 1, directory_name)
                self.assertEqual(len(tables), 1, directory_name)
                self.assertGreater(pngs[0].stat().st_size, 20_000)
                self.assertGreater(pdfs[0].stat().st_size, 5_000)
                self.assertTrue(pdfs[0].read_bytes().startswith(b"%PDF"))
                with Image.open(pngs[0]) as image:
                    self.assertGreaterEqual(image.size[0], 2400)
                    self.assertGreaterEqual(image.size[1], 1800)
                    self.assertAlmostEqual(float(image.info["dpi"][0]), 300.0, delta=1.0)

            concept_a = json.loads(
                (output / CONCEPT_DIRS[0] / "metadata.json").read_text()
            )
            self.assertTrue(concept_a["excluded_from_scientific_metrics"])
            self.assertEqual(
                concept_a["synthetic_data_status"],
                "synthetic_mechanics_layout_demo",
            )

            decision = pd.read_csv(
                output
                / "concept_D_candidate_decision"
                / "concept_D_candidate_decision_tidy_data.csv"
            )
            unknown = decision["viability_prediction_status"].str.startswith(
                "unknown_"
            )
            self.assertTrue(decision.loc[unknown, "public_viability_mean"].isna().all())
            self.assertIn("empirical_combination_pass_probability", decision.columns)
            self.assertNotIn("intact_patch_pass_probability", decision.columns)

            timeline = pd.read_csv(
                output
                / "concept_B_campaign_timeline"
                / "concept_B_campaign_timeline_tidy_data.csv"
            )
            self.assertIn("provenance", timeline.columns)
            self.assertIn("phase", timeline.columns)
            mechanics = timeline.loc[
                timeline["series"].eq("mechanical_measurement_count"), "value"
            ]
            self.assertEqual(float(mechanics.sum()), 0.0)
            caption = (
                output / "concept_E_publication_triptych" / "caption.txt"
            ).read_text()
            self.assertIn("Hypervolume is withheld", caption)


if __name__ == "__main__":
    unittest.main()
