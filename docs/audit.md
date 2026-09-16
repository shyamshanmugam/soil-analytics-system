# Repository audit

## Findings

The initial commit contained only `README.md` with the text `oubica`. There were no
Python files, notebooks, data files, frontend files, tests, model artefacts, or
configuration files. Consequently, there was no Week 3 preprocessing or CNN code to
reuse, no duplicated implementation to remove, and no experimental result to report.

## Architecture decision

The baseline is split into:

1. `src/structured_ml`: validated laboratory values and a transparent rule-based
   baseline that can later be replaced by a gradient-boosted model.
2. `src/cnn`: an explicit integration point returning `not_trained` until a CNN is
   trained on the documented soil classes.
3. `src/recommendations`: deterministic, inspectable advisory rules.
4. `backend/app`: FastAPI validation and HTTP endpoints.
5. `frontend`: a later farmer-facing client consuming the API.

This separation prevents an image classifier from being presented as a nutrient
predictor, which the available image dataset cannot support.
