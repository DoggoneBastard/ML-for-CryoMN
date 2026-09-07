from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = PROJECT_ROOT / "src" / "08_multi_objective"
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from helper.config import (  # noqa: E402
    ConfigValidationError,
    load_availability_config,
    load_endpoints_config,
    load_evaluation_config,
    load_ingredients_config,
    load_optimization_config,
    validate_availability_config,
    validate_endpoints_config,
    validate_evaluation_config,
    validate_ingredients_config,
    validate_optimization_config,
)
from helper.paths import portable_source_path  # noqa: E402


class V2ConfigValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ingredients = load_ingredients_config()
        cls.endpoints = load_endpoints_config()
        cls.optimization = load_optimization_config()
        cls.availability = load_availability_config()
        cls.evaluation = load_evaluation_config()

    def test_current_configuration_is_valid(self) -> None:
        validate_ingredients_config(self.ingredients)
        validate_endpoints_config(self.endpoints)
        validate_optimization_config(self.optimization)
        validate_availability_config(self.availability, self.ingredients)
        validate_evaluation_config(self.evaluation)

    def test_reversed_ingredient_bounds_are_rejected(self) -> None:
        config = deepcopy(self.ingredients)
        config["ingredients"][0]["lower_bound"] = 2.0
        config["ingredients"][0]["upper_bound"] = 1.0
        with self.assertRaisesRegex(ConfigValidationError, "must not exceed"):
            validate_ingredients_config(config)

    def test_unknown_availability_ingredient_is_rejected(self) -> None:
        config = {"temporarily_unavailable_feature_names": ["not_an_ingredient"]}
        with self.assertRaisesRegex(ConfigValidationError, "unknown ingredient"):
            validate_availability_config(config, self.ingredients)

    def test_missing_required_objective_is_rejected(self) -> None:
        config = deepcopy(self.endpoints)
        config["primary_objectives"] = config["primary_objectives"][:1]
        with self.assertRaisesRegex(ConfigValidationError, "missing required endpoint"):
            validate_endpoints_config(config)

    def test_invalid_phase_mode_is_rejected(self) -> None:
        config = deepcopy(self.optimization)
        config["phase_mode"] = "unrecognized"
        with self.assertRaisesRegex(ConfigValidationError, "phase_mode"):
            validate_optimization_config(config)

    def test_non_numeric_reference_point_is_rejected(self) -> None:
        config = deepcopy(self.optimization)
        config["selection"]["reference_point"]["viability_percent"] = "low"
        with self.assertRaisesRegex(ConfigValidationError, "must be numeric"):
            validate_optimization_config(config)

    def test_mechanical_capacity_cannot_exceed_slate(self) -> None:
        config = deepcopy(self.optimization)
        config["mechanics_transition"]["bootstrap"]["mechanical_capacity"] = 13
        with self.assertRaisesRegex(ConfigValidationError, "must not exceed"):
            validate_optimization_config(config)

    def test_full_gate_cannot_be_lower_than_hybrid_gate(self) -> None:
        config = deepcopy(self.optimization)
        config["mechanics_transition"]["full_gate"]["min_paired_observations"] = 6
        with self.assertRaisesRegex(ConfigValidationError, "received 6 < 8"):
            validate_optimization_config(config)

    def test_unknown_evaluation_cohort_is_rejected(self) -> None:
        config = deepcopy(self.evaluation)
        config["round_provenance"]["ROUND_010"] = "mystery_cohort"
        with self.assertRaisesRegex(ConfigValidationError, "unrecognized cohort"):
            validate_evaluation_config(config)

    def test_portable_source_paths_do_not_require_existing_files(self) -> None:
        self.assertEqual(
            portable_source_path("results/multi_objective_v2/future.csv"),
            "results/multi_objective_v2/future.csv",
        )
        self.assertEqual(
            portable_source_path("/private/tmp/external-future.csv"),
            "/private/tmp/external-future.csv",
        )


if __name__ == "__main__":
    unittest.main()
