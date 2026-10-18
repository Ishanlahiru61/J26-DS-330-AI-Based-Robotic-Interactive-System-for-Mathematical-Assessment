# Child Behaviour and Motor-Control Analysis — Member 3

This module analyses hand landmarks, finger states, movement sequences, joint
angles, head pose, and coarse gaze locally. It reports confidence, duration, and
tracking quality and uses explicit uncertain/unknown states when visibility is
insufficient.

Layering: `api.py -> service.py -> domain/ -> tracking, pose, gaze, feature,
classification, and model adapters`.
