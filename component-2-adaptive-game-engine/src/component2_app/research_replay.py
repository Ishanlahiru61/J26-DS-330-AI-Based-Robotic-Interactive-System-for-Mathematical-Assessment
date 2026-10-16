"""Observed-task replay: probabilities influence a support decision, not local game ranking."""
from pathlib import Path
import math


def support_decision(probability: float) -> dict:
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('Invalid calibrated probability')
    action = ('SCAFFOLDED_PRACTICE' if probability < .60 else
              'STANDARD_PRACTICE' if probability <= .85 else 'CONSIDER_NEXT_CHALLENGE')
    return {'action': action, 'decision_mode': 'RESEARCH_MODEL_INFORMED_SUPPORT',
            'predicted_probability': float(probability),
            'policy_version': 'REPLAY_SUPPORT_HEURISTIC_v1',
            'rationale': 'p<0.60: support; 0.60<=p<=0.85: standard; p>0.85: consider challenge',
            'validation_status': 'illustrative engineering policy; not outcome-validated',
            'local_activity_id': None}


def replay(workspace: str | Path, limit: int = 8) -> dict:
    import duckdb
    import numpy as np
    from recommendation_engine.model_scorer import FrozenModelScorer
    if type(limit) is not int or not 1 <= limit <= 32:
        raise ValueError('Replay limit must be 1–32')
    root = Path(workspace).resolve()
    scorer = FrozenModelScorer.from_workspace(root)
    predictions = root / '03_Experimental_Modeling/data/prepared/predictions_H2_GB_RT_OFF_test_platt_retained.parquet'
    history = root / '02_Chronological_Learner_History/data/processed/ASSISTments_2012_2013_history_features.parquet'
    # Read only deterministic saved test observations; never rebuild the full dataset.
    with duckdb.connect() as con:
        saved = con.execute('SELECT * FROM read_parquet(?) ORDER BY user_id,problem_log_id LIMIT ?', [str(predictions), limit]).df()
        con.register('replay_ids', saved[['problem_log_id']])
        rows = con.execute('SELECT h.* FROM read_parquet(?) h JOIN replay_ids USING(problem_log_id) ORDER BY h.user_id,h.learner_interaction_index,h.problem_log_id', [str(history)]).df()
    if len(rows) != limit or set(rows['problem_log_id']) != set(saved['problem_log_id']):
        raise ValueError('Frozen observation identities do not align')
    saved = saved.set_index('problem_log_id').loc[rows['problem_log_id']].reset_index()
    probabilities = scorer.score_raw_research(rows, provenance='FROZEN_ASSISTMENTS_HISTORY')
    changed_outcomes = rows.copy()
    changed_outcomes['target_first_attempt_correct'] = 1 - changed_outcomes['target_first_attempt_correct']
    check = scorer.score_raw_research(changed_outcomes, provenance='FROZEN_ASSISTMENTS_HISTORY')
    if not np.array_equal(probabilities, check):
        raise ValueError('Current outcomes affected predictions')
    differences = np.abs(probabilities - saved['predicted_probability'].to_numpy())
    if not np.all(differences < 2e-6):
        raise ValueError('Fresh calibrated inference differs from frozen reference')
    decisions = []
    for i, p in enumerate(probabilities):
        row = rows.iloc[i]
        decisions.append({
            'observation_id': int(row['problem_log_id']), 'research_learner_id': int(row['user_id']),
            'timestamp': str(row['start_time_parsed']), 'history_count': int(row['prior_interaction_count']),
            'observed_skill_id': str(row['skill_id']), 'observed_template_id': str(row['template_id']),
            'prediction_then_decision': support_decision(float(p)),
            'saved_reference_probability': float(saved.iloc[i]['predicted_probability']),
            'absolute_difference': float(differences[i]),
            # Outcome is revealed only after input transformation and decision.
            'subsequently_observed_correctness': int(saved.iloc[i]['y_true']),
        })
    return {'mode': 'FROZEN_RESEARCH_OBSERVATION_REPLAY', 'model': 'H2_GB_RT_OFF_v1 + Platt',
            'raw_feature_count': len(scorer.raw_features), 'raw_features': scorer.raw_features,
            'removed_rt_fields': sorted(__import__('recommendation_engine.model_scorer', fromlist=['RT_FIELDS']).RT_FIELDS),
            'transformed_feature_width': len(scorer.selected_columns),
            'current_outcome_invariance_check': 'PASS',
            'artifacts_verified': scorer.artifacts, 'decisions': decisions,
            'claim_boundary': 'Observed-task predictions drive illustrative support actions. No counterfactual candidate ranking, local mapping or causal benefit is claimed.'}
