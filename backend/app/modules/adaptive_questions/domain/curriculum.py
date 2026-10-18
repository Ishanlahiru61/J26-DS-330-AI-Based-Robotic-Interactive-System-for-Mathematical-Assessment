"""Conservative question-level policy from the saved NIE audit, not attainment norms."""
GRADE_MIN = {
    ('ORDERING', 'Tier B'): 2,
    ('PLACE_VALUE', 'Tier A'): 2, ('PLACE_VALUE', 'Tier B'): 2,
    ('ADDITION', 'Tier B'): 2, ('SUBTRACTION', 'Tier B'): 2,
    ('MULTIPLICATION', 'Tier A'): 3, ('MULTIPLICATION', 'Tier B'): 5,
    ('DIVISION', 'Tier A'): 3, ('DIVISION', 'Tier B'): 4,
}
SOURCES = {
    'COUNTING': 'NIE G1 PDF 36,44,56', 'COMPARISON': 'NIE G1 PDF 21',
    'ORDERING': 'NIE G2 PDF 15; G1 step-one foundational adaptation',
    'PLACE_VALUE': 'NIE G2 PDF 15; G3 PDF 104',
    'ADDITION': 'NIE G1 PDF 44 (initial sum <=6); G2 PDF 15',
    'SUBTRACTION': 'NIE topic established; G1 exact ceiling requires verification',
    'MULTIPLICATION': 'NIE G3 PDF 131 (factor 2); G5 PDF 15,18 (wider factors)',
    'DIVISION': 'NIE G3 PDF 53,133 (by 2); G4 PDF 13 (by 3,4)',
}

def grade_support(game: str, tier: str, grade: int) -> bool:
    return (type(grade) is int and 1 <= grade <= 5 and game in SOURCES
            and tier in {'Tier A', 'Tier B'} and grade >= GRADE_MIN.get((game, tier), 1))

def curriculum_details(game: str, tier: str, grade: int) -> dict:
    return {'grade': grade, 'eligible': grade_support(game, tier, grade),
            'evidence': SOURCES[game], 'policy_version': 'NIE_CONSERVATIVE_PP1_v2',
            'verification_status': ('REQUIRES_CURRICULUM_VERIFICATION' if
                (game == 'SUBTRACTION' and grade == 1) or grade >= 4
                else 'LOCAL_NIE_AUDIT_WITH_PROJECT_TASK_ADAPTATION'),
            'scope': 'foundational task sample; not full grade attainment'}
