#!/usr/bin/env python3
"""
Extra Redis structures for the UI demo. Does NOT change the dataset in demo.py.

  python extras.py        # run once after `python demo.py load`

Creates:
  sug:care     autocomplete dictionary          (FT.SUGADD / FT.SUGGET)
  synonyms     lay-term -> clinical-term groups  (FT.SYNUPDATE on idx:care)
  idx:route    intent router: example phrases as vectors (FT.SEARCH KNN)
"""
import os, re
import demo

r, IDX = demo.r, demo.IDX
SUG = "sug:care"
ROUTE_IDX = "idx:route"

# Tunable thresholds (cosine distance, 0 = identical). Shown in the UI so you can tune live.
ROUTE_T = float(os.environ.get("ROUTE_T", 0.60))      # max distance to accept an intent
EMERG_T = float(os.environ.get("EMERG_T", 0.50))      # max distance to trigger emergency by meaning
NOANSWER_T = float(os.environ.get("NOANSWER_T", 0.72))  # best vector distance above this = no grounded answer

# ---------------------------------------------------------------- synonyms (lay terms patients type)
SYNONYMS = {
    "syn_doctor": ["doctor", "dr", "physician", "provider"],
    "syn_kids": ["kids", "kid", "child", "children", "pediatric", "pediatrics", "pediatrician", "baby"],
    "syn_heart": ["heart", "cardiac", "cardiology", "cardiologist", "cardio"],
    "syn_skin": ["skin", "dermatology", "dermatologist", "derm"],
    "syn_women": ["obgyn", "gynecology", "gynecologist", "obstetrics", "womens", "pregnancy", "prenatal"],
    "syn_bone": ["bone", "bones", "joint", "joints", "orthopedic", "orthopedics", "ortho"],
    "syn_brain": ["brain", "nerve", "neurology", "neurologist", "neuro"],
    "syn_stomach": ["stomach", "tummy", "belly", "gi", "gastroenterology", "gastroenterologist", "digestive"],
    "syn_cancer": ["cancer", "oncology", "oncologist", "tumor"],
    "syn_pt": ["rehab", "therapy", "pt", "physio"],
}

# ---------------------------------------------------------------- intent router examples
ROUTES = {
    "emergency": ["chest pain and I can't breathe", "I think I'm having a heart attack",
                  "stroke symptoms face drooping slurred speech", "severe bleeding that won't stop",
                  "someone is unconscious", "difficulty breathing right now"],
    "schedule": ["schedule an appointment with Dr", "book a visit with my doctor", "make an appointment",
                 "next available appointment", "can I see a doctor this week", "schedule Dr Peters"],
    "urgent": ["urgent care open now", "sprained my ankle", "fever and sore throat", "cut that needs stitches",
               "walk in clinic near me", "flu symptoms"],
    "find_care": ["I need a doctor for my knee", "find a specialist", "my back hurts", "rash that won't go away",
                  "I can't sleep at night", "who treats migraines", "doctor for my child"],
    "location": ["directions to the hospital", "where is the clinic", "parking at the medical center",
                 "what are your hours", "address of the health pavilion"],
    "jobs": ["nursing jobs", "careers at the hospital", "are you hiring", "job openings"],
}
EMERGENCY_RE = re.compile(r"chest pain|can'?t breathe|cannot breathe|trouble breathing|not breathing|stroke|"
                          r"heart attack|unconscious|severe bleeding|overdose|\b911\b", re.I)


def load_extras():
    # autocomplete
    r.delete(SUG)
    seen = set()
    for d in demo.DOCS:
        for s, score in [(d["title"], 3), (d["title"].replace("Dr. ", ""), 2), (d["specialty"], 1)]:
            if s and s.lower() not in seen:
                seen.add(s.lower()); r.execute_command("FT.SUGADD", SUG, s, score)
    for s in ["schedule an appointment", "urgent care near me", "primary care doctor", "kids doctor",
              "heart doctor", "skin doctor", "knee pain", "back pain", "mammogram", "telehealth visit"]:
        if s not in seen:
            r.execute_command("FT.SUGADD", SUG, s, 1)

    # synonyms (without SKIPINITIALSCAN, existing docs are re-scanned)
    for gid, terms in SYNONYMS.items():
        r.execute_command("FT.SYNUPDATE", IDX, gid, *terms)

    # intent router index
    try:
        r.execute_command("FT.DROPINDEX", ROUTE_IDX, "DD")
    except Exception:
        pass
    r.execute_command("FT.CREATE", ROUTE_IDX, "ON", "HASH", "PREFIX", "1", "route:", "SCHEMA",
                      "route", "TAG", "example", "TEXT",
                      "vec", "VECTOR", "FLAT", "6", "TYPE", "FLOAT32", "DIM", str(demo.DIM), "DISTANCE_METRIC", "COSINE")
    pipe = r.pipeline(); i = 0
    for route, exs in ROUTES.items():
        for ex in exs:
            pipe.hset(f"route:{i}", mapping={"route": route, "example": ex, "vec": demo.embed(ex)}); i += 1
    pipe.execute()
    print(f"suggestions: {len(seen)}+, synonym groups: {len(SYNONYMS)}, intent examples: {i}")


def classify(q, vec):
    """Return dict(route, dist, example, reason). Deterministic emergency backstop + vector router."""
    res = r.execute_command("FT.SEARCH", ROUTE_IDX, "*=>[KNN 3 @vec $v AS dist]", "PARAMS", "2", "v", vec,
                            "SORTBY", "dist", "RETURN", "3", "route", "example", "dist", "DIALECT", "2")
    rows = demo.parse(res)
    top = rows[0] if rows else {"route": "general", "example": "", "dist": "1"}
    dist = float(top.get("dist", 1))
    if EMERGENCY_RE.search(q):
        return dict(route="emergency", dist=dist, example=top.get("example"), reason="safety keyword rule")
    if top["route"] == "emergency" and dist <= EMERG_T:
        return dict(route="emergency", dist=dist, example=top["example"], reason="semantic match")
    if top["route"] != "emergency" and dist <= ROUTE_T:
        return dict(route=top["route"], dist=dist, example=top["example"], reason="semantic match")
    return dict(route="general", dist=dist, example=top.get("example"), reason="no confident intent")


if __name__ == "__main__":
    load_extras()
