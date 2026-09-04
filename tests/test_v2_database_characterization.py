from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = PROJECT_ROOT / "src" / "08_multi_objective"
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from helper.config import load_optimization_config  # noqa: E402
from helper.paths import (  # noqa: E402
    FORMULATIONS_PATH,
    LEGACY_LITERATURE_PATH,
    LEGACY_VALIDATION_PATH,
    OBSERVATIONS_PATH,
)
from helper.registry import load_registry  # noqa: E402
from helper.transfer import build_v2_tables_from_legacy  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class V2DatabaseCharacterizationTests(unittest.TestCase):
    def test_canonical_long_form_database_contract(self) -> None:
        formulations = pd.read_csv(FORMULATIONS_PATH)
        observations = pd.read_csv(OBSERVATIONS_PATH)
        self.assertEqual(formulations.shape, (392, 29))
        self.assertEqual(observations.shape, (796, 11))
        self.assertEqual(
            observations["endpoint"].value_counts().to_dict(),
            {"viability_percent": 628, "intact_patch_formation_pass": 168},
        )
        self.assertEqual(
            observations["source_type"].value_counts().to_dict(),
            {
                "wetlab_feedback": 492,
                "legacy_literature": 198,
                "legacy_wetlab": 106,
            },
        )
        self.assertTrue(formulations["formulation_id"].is_unique)
        self.assertTrue(observations["observation_id"].is_unique)
        self.assertFalse(observations["unit"].fillna("").eq("").any())

    def test_stage_one_transfer_changes_only_new_provenance_strings(self) -> None:
        before_hashes = {
            path: _sha256(path)
            for path in (LEGACY_LITERATURE_PATH, LEGACY_VALIDATION_PATH)
        }
        rebuilt_formulations, rebuilt_observations = build_v2_tables_from_legacy(
            LEGACY_LITERATURE_PATH,
            LEGACY_VALIDATION_PATH,
            load_registry(),
            load_optimization_config(),
        )
        canonical_formulations = pd.read_csv(FORMULATIONS_PATH)
        canonical_observations = pd.read_csv(OBSERVATIONS_PATH)
        canonical_formulations = canonical_formulations.loc[
            canonical_formulations["formulation_id"].isin(
                rebuilt_formulations["formulation_id"]
            )
        ].sort_values("formulation_id").reset_index(drop=True)
        rebuilt_formulations = rebuilt_formulations.sort_values(
            "formulation_id"
        ).reset_index(drop=True)
        canonical_observations = canonical_observations.loc[
            canonical_observations["observation_id"].isin(
                rebuilt_observations["observation_id"]
            )
        ].sort_values("observation_id").reset_index(drop=True)
        rebuilt_observations = rebuilt_observations.sort_values(
            "observation_id"
        ).reset_index(drop=True)
        pd.testing.assert_frame_equal(
            canonical_formulations,
            rebuilt_formulations,
            check_dtype=False,
        )
        scientific_columns = [
            column
            for column in canonical_observations.columns
            if column != "source_file"
        ]
        pd.testing.assert_frame_equal(
            canonical_observations[scientific_columns],
            rebuilt_observations[scientific_columns],
            check_dtype=False,
        )
        self.assertTrue(
            rebuilt_observations["source_file"].str.startswith("data/").all()
        )
        self.assertEqual(
            before_hashes,
            {
                path: _sha256(path)
                for path in (LEGACY_LITERATURE_PATH, LEGACY_VALIDATION_PATH)
            },
        )


if __name__ == "__main__":
    unittest.main()
