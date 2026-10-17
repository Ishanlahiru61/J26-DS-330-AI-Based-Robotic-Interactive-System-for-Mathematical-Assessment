from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st
from component2_app import Component2Service
from views import child_view, researcher_view, replay_view

st.set_page_config(page_title="Component 2 Adaptive Cube Games", page_icon="🎲", layout="wide")


def get_service():
    override = os.environ.get("COMPONENT2_DB_PATH")
    service = Component2Service(override) if override else Component2Service()
    if not service.db.learners():
        service.seed_demo_learners()
    return service


service = get_service()
mode = st.sidebar.radio("View", ("Researcher", "Child", "Research replay"))
st.sidebar.caption("PP1 prototype · experimental local learner-state model · nonclinical")
if mode == "Researcher":
    researcher_view.render(service)
elif mode == "Child":
    child_view.render(service)
else:
    replay_view.render()
