# Feature engineering

Runtime transforms build the frozen 25-feature, strictly prior learner-history
row. The live transferable schema excludes current-response, identifier, and
response-time fields; hints and C1 quality remain descriptive evidence. Offline
fitting belongs in `backend/training/adaptive_questions/`.
