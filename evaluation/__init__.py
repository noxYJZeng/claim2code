"""Claim2Code evaluation harness — package marker.

The eval scripts under ``evaluation/`` (``baselines/``,
``evaluate_retrieval.py``) import the live retrieval scorers
from :mod:`claim2code_api`. The :file:`evaluate_retrieval.py`
runner adjusts ``sys.path`` to include ``apps/api`` before
the baselines are loaded — see ``_setup_path`` in that file.
"""