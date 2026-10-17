# Adaptive Question Recommendation — Member 2

This module uses chronological assessment history to estimate first-attempt
success probability and select the next task and difficulty. Recommendations
must consider correctness, time, attempts, hints, recent performance, age, and
curriculum constraints.

Layering: `api.py -> service.py -> domain/ -> feature, prediction, recommendation,
and model adapters`.
