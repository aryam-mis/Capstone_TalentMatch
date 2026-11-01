# tests/test_explanations.py
# Checks for the cold-start explanations and the senior-title rule, without data files
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from profile_recommender import JUNIOR_YEARS, SENIOR_RE, ProfileRecommender  # noqa: E402
from why_skills import similar_role  # noqa: E402


def test_similar_role_picks_shared_occupation_words():
    assert similar_role("Scheduler", ["Manager", "Dispatcher/Scheduler"]) == "Dispatcher/Scheduler"
    assert similar_role("SQL Developer", ["Data Analyst Intern", "Python Developer"]) == "Python Developer"


def test_similar_role_ignores_generic_words():
    # "associate" alone is not an occupation, so no role is claimed
    assert similar_role("Sales Associate", ["Customer Service Associate"]) is None
    assert similar_role("Front Desk Receptionist", []) is None


def test_senior_titles_are_detected():
    for t in ["Sr. Software Engineer", "Senior Analyst", "BI Development Lead", "DEVELOPER IV", "Architect, Data"]:
        assert re.search(SENIOR_RE, t), t
    for t in ["SQL Developer", "Software Engineer", "Junior Data Analyst", "Staffing Coordinator"]:
        assert not re.search(SENIOR_RE, t), t
    assert JUNIOR_YEARS == 3


def test_profile_text_adds_skills_only_when_given():
    base = ProfileRecommender.profile_text("bachelor's", "ai", 0, [])
    assert base.rstrip().endswith("past roles:")
    with_skills = ProfileRecommender.profile_text("bachelor's", "ai", 0, [], "Python, SQL")
    assert with_skills == "degree: bachelor's. major: ai. experience: 0 years. skills: python, sql"
