import json
import shutil
import tempfile
import unittest
from pathlib import Path

from backend.app.modules.adaptive_questions.recommendation.coverage import GAME_TO_SKILL, choose_adaptive_focus
from backend.app.modules.adaptive_questions.feature_engineering.learner_state import make_learner_state
from backend.app.modules.adaptive_questions.prediction.predictor import PredictorRouter, TransferableTrainedPredictor


class TransferablePredictorTests(unittest.TestCase):
    def test_artifact_loads_and_scores_strict_prior_state(self):
        state = make_learner_state(
            learner_id="L", current_grade=3, prior_interaction_count=3,
            prior_correct_count=2, recent_outcomes=(1, 0, 1),
            recent_skill_outcomes_by_skill={"Addition": (1, 0, 1)},
            prior_skill_exposures={"Addition": 3}, prior_skill_correct={"Addition": 2},
            interactions_since_last_correct=0,
        )
        result = TransferableTrainedPredictor().predict(state, "Addition")
        self.assertTrue(result["valid"])
        self.assertEqual(result["source"], "TRAINED_ML")
        self.assertEqual(result["model_version"], "TRANSFER_H2_GB_RT_OFF_v1")
        self.assertGreaterEqual(result["success_probability"], 0)
        self.assertLessEqual(result["success_probability"], 1)
        forbidden = {"correct", "target", "skill_id", "problem_id", "assistment_id", "template_id", "tutor_mode"}
        self.assertFalse(forbidden & set(result["feature_values"]))

    def test_cold_start_and_missing_artifact_use_explicit_fallback_or_abstention(self):
        cold = make_learner_state(learner_id="L", current_grade=3)
        self.assertEqual(PredictorRouter().predict(cold, "Addition")["source"], "ABSTAIN")
        with tempfile.TemporaryDirectory() as folder:
            missing = PredictorRouter(TransferableTrainedPredictor(folder))
            state = make_learner_state(
                learner_id="L", current_grade=3, prior_interaction_count=2,
                prior_correct_count=1, recent_outcomes=(1, 0),
                recent_skill_outcomes_by_skill={"Addition": (1, 0)},
                prior_skill_exposures={"Addition": 2}, prior_skill_correct={"Addition": 1},
            )
            result = missing.predict(state, "Addition")
            self.assertEqual(result["source"], "BETA_BERNOULLI_FALLBACK")
            self.assertEqual(result["model_version"], "LOCAL_BETA_BERNOULLI_v1")
            self.assertIn("ARTIFACT_LOAD_FAILED", result["primary_abstention_reason"])

    def test_frozen_metadata_and_evaluation_are_self_consistent(self):
        root = Path(__file__).resolve().parents[2] / "app" / "modules" / "adaptive_questions" / "models" / "transferable"
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        evaluation = json.loads((root / "evaluation.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["current_response_features"], [])
        self.assertEqual(metadata["identifier_features"], [])
        self.assertEqual(evaluation["test_native"]["n_rows"], 937043)
        self.assertFalse(evaluation["calibration_retained"])
        self.assertGreater(evaluation["test_native"]["roc_auc"], evaluation["beta_bernoulli_test_supported_rows"]["roc_auc"])

    def test_changed_artifact_is_rejected_by_manifest_integrity_check(self):
        source = Path(__file__).resolve().parents[2] / "app" / "modules" / "adaptive_questions" / "models" / "transferable"
        with tempfile.TemporaryDirectory() as folder:
            copied = Path(folder) / "transferable"
            shutil.copytree(source, copied)
            (copied / "evaluation.json").write_text("{}", encoding="utf-8")
            predictor = TransferableTrainedPredictor(copied)
            self.assertIn("ARTIFACT_LOAD_FAILED", predictor.load_error)


class PredictionSensitivePolicyTests(unittest.TestCase):
    def setUp(self):
        self.profiles = [
            {"game_type": game, "valid_interactions": 4 if game == "COUNTING" else 0,
             "last_assessed_at": "2026-01-01" if game == "COUNTING" else None}
            for game in ("COUNTING", "COMPARISON", "ORDERING")
        ]
        self.candidates = [
            {"game_type": game, "project_skill": GAME_TO_SKILL[game], "tier": "Tier A", "activity_id": game}
            for game in ("COUNTING", "COMPARISON", "ORDERING")
        ]
        self.state = make_learner_state(
            learner_id="L", current_grade=3, prior_interaction_count=3,
            prior_correct_count=2, current_skill_context=GAME_TO_SKILL["COUNTING"],
            recent_outcomes=(1, 1, 0), recent_skill_outcomes=(1, 1, 0),
            recent_skill_outcomes_by_skill={GAME_TO_SKILL["COUNTING"]: (1, 1, 0)},
            prior_skill_exposures={GAME_TO_SKILL["COUNTING"]: 3},
            prior_skill_correct={GAME_TO_SKILL["COUNTING"]: 2}, last_tier="Tier B",
        )

    @staticmethod
    def predictions(comparison, ordering):
        return {
            GAME_TO_SKILL["COUNTING"]: {"success_probability": .6, "source": "TRAINED_ML", "valid": True},
            GAME_TO_SKILL["COMPARISON"]: {"success_probability": comparison, "source": "TRAINED_ML", "valid": True},
            GAME_TO_SKILL["ORDERING"]: {"success_probability": ordering, "source": "TRAINED_ML", "valid": True},
        }

    def test_changing_only_predictions_changes_selected_game(self):
        recent = [{"game_type": "COUNTING", "correct": 0, "timestamp": "2026-01-01"}]
        first = choose_adaptive_focus(self.profiles, self.candidates, recent, self.state, self.predictions(.65, .95))
        second = choose_adaptive_focus(self.profiles, self.candidates, recent, self.state, self.predictions(.95, .65))
        self.assertEqual(first["game_type"], "COMPARISON")
        self.assertEqual(second["game_type"], "ORDERING")
        self.assertEqual(first["reason_code"], "MODEL_RANKED_COVERAGE_CANDIDATE")

    def test_prediction_threshold_changes_tier_action_without_bypassing_readiness(self):
        skill = GAME_TO_SKILL["COUNTING"]
        state = make_learner_state(
            learner_id="L", current_grade=3, prior_interaction_count=3, prior_correct_count=3,
            current_skill_context=skill, recent_outcomes=(1, 1, 1), recent_skill_outcomes=(1, 1, 1),
            recent_skill_outcomes_by_skill={skill: (1, 1, 1)}, prior_skill_exposures={skill: 3},
            prior_skill_correct={skill: 3}, tier_a_exposures={skill: 3}, tier_a_correct={skill: 3},
            last_tier="Tier A",
        )
        candidates = [
            {"game_type": "COUNTING", "project_skill": skill, "tier": tier, "activity_id": tier}
            for tier in ("Tier A", "Tier B")
        ]
        profile = [{"game_type": "COUNTING", "valid_interactions": 3, "last_assessed_at": "2026-01-01"}]
        recent = [{"game_type": "COUNTING", "correct": 1, "timestamp": str(i)} for i in range(3)]
        high = choose_adaptive_focus(profile, candidates, recent, state, {skill: {"success_probability": .8, "source": "TRAINED_ML", "valid": True}})
        low = choose_adaptive_focus(profile, candidates, recent, state, {skill: {"success_probability": .3, "source": "TRAINED_ML", "valid": True}})
        self.assertEqual(high["decision_type"], "ADVANCE")
        self.assertEqual(low["decision_type"], "REINFORCE")


if __name__ == "__main__":
    unittest.main()
