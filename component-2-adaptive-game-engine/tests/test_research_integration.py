import os
import unittest
from pathlib import Path

from component2_app.research_replay import support_decision, replay


class ResearchPolicyTests(unittest.TestCase):
    def test_probability_genuinely_changes_action(self):
        self.assertEqual(support_decision(.3)['action'],'SCAFFOLDED_PRACTICE')
        self.assertEqual(support_decision(.7)['action'],'STANDARD_PRACTICE')
        self.assertEqual(support_decision(.95)['action'],'CONSIDER_NEXT_CHALLENGE')
        self.assertIsNone(support_decision(.7)['local_activity_id'])

    def test_invalid_probabilities_rejected(self):
        for p in (-1,2,float('nan'),float('inf')):
            with self.assertRaises(ValueError): support_decision(p)

    def test_missing_artifacts_fail_explicitly(self):
        import tempfile
        from recommendation_engine.model_scorer import FrozenModelScorer
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
            FrozenModelScorer.from_workspace(folder)


@unittest.skipUnless(os.environ.get('C2_RESEARCH_WORKSPACE'),'External frozen research workspace not configured')
class FrozenIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from recommendation_engine.model_scorer import FrozenModelScorer
        cls.scorer=FrozenModelScorer.from_workspace(os.environ['C2_RESEARCH_WORKSPACE'])

    def test_schema_dimensions_and_rt_off_fields(self):
        self.assertEqual(len(self.scorer.raw_features),42)
        self.assertEqual(len(self.scorer.selected_columns),self.scorer.model.n_features_in_)
        self.assertIn('time_since_previous_seconds',self.scorer.raw_features)

    def test_feature_and_provenance_rejection(self):
        import pandas as pd
        with self.assertRaises(ValueError): self.scorer.score_raw_research(pd.DataFrame(),provenance='LOCAL_GAME')
        with self.assertRaises(ValueError): self.scorer.score_raw_research(pd.DataFrame(),provenance='FROZEN_ASSISTMENTS_HISTORY')
        import numpy as np
        with self.assertRaises(ValueError): self.scorer.score_pretransformed(np.zeros((1,1)))

    def test_fresh_inference_matches_saved_reference_and_drives_decisions(self):
        report=replay(os.environ['C2_RESEARCH_WORKSPACE'],8)
        self.assertEqual(len(report['decisions']),8)
        self.assertEqual(report['current_outcome_invariance_check'],'PASS')
        for row in report['decisions']:
            self.assertLess(row['absolute_difference'],2e-6)
            prediction=row['prediction_then_decision']
            self.assertEqual(prediction['action'],support_decision(prediction['predicted_probability'])['action'])


if __name__=='__main__': unittest.main()
