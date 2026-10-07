#!/usr/bin/env python3
"""
Healthcare site-search demo for Acme Health: keyword + fuzzy + vector + hybrid on Redis Query Engine.
All data is SYNTHETIC (no PHI, no real providers).

  pip install redis sentence-transformers numpy
  export REDIS_URL="redis://default:<password>@<host>:<port>"
  python demo.py load            # create index + load data (once, before the call)
  python demo.py run             # guided run of all demo queries with latencies
  python demo.py q "my knee hurts when I climb stairs"   # ad-hoc, shows all modes side by side
  python demo.py availability    # live-ish availability update shows up instantly in search

Requires Redis 8.4+ / Redis Cloud with Query Engine for FT.HYBRID. Falls back to
KNN-with-prefilter (works on all versions) if FT.HYBRID is unavailable.
"""
import os, sys, time, struct, json
import numpy as np
import redis

URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
r = redis.Redis.from_url(URL, decode_responses=False)
IDX, DIM = "idx:care", 384
_model = None


def embed(text: str) -> bytes:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    v = _model.encode(text, normalize_embeddings=True).astype(np.float32)
    return v.tobytes()


# ---------------------------------------------------------------- synthetic data
# type: provider | service | condition | location
DOCS = [
 # providers
 dict(id="p1", type="provider", title="Dr. Madeline Peters", specialty="Orthopedic Surgery", location="Dayton", lon=-84.19, lat=39.76, accepting="yes", next_slot=1,
      body="Orthopedic surgeon specializing in knee and hip replacement, sports injuries, ACL repair, arthritis and joint pain."),
 dict(id="p2", type="provider", title="Dr. Samuel Okafor", specialty="Cardiology", location="Springfield-Demo", lon=-84.17, lat=39.69, accepting="yes", next_slot=3,
      body="Cardiologist treating chest pain, high blood pressure, heart failure, atrial fibrillation and heart disease prevention."),
 dict(id="p3", type="provider", title="Dr. Priya Raman", specialty="Pediatrics", location="Beavercreek", lon=-84.06, lat=39.71, accepting="yes", next_slot=0,
      body="Pediatrician providing well-child visits, vaccinations, childhood asthma, ear infections and developmental screening."),
 dict(id="p4", type="provider", title="Dr. Elena Vasquez", specialty="Obstetrics & Gynecology", location="Centerville", lon=-84.16, lat=39.63, accepting="yes", next_slot=2,
      body="OB/GYN offering prenatal care, high-risk pregnancy, annual women's exams, menopause care and minimally invasive surgery."),
 dict(id="p5", type="provider", title="Dr. Marcus Lee", specialty="Gastroenterology", location="Dayton", lon=-84.19, lat=39.76, accepting="no", next_slot=21,
      body="Gastroenterologist for acid reflux, IBS, colonoscopy screening, Crohn's disease and stomach pain."),
 dict(id="p6", type="provider", title="Dr. Anita Shah", specialty="Dermatology", location="Beavercreek", lon=-84.06, lat=39.71, accepting="yes", next_slot=5,
      body="Dermatologist treating acne, eczema, psoriasis, rashes, skin cancer screening and mole checks."),
 dict(id="p7", type="provider", title="Dr. Robert Kim", specialty="Primary Care", location="Springboro", lon=-84.23, lat=39.56, accepting="yes", next_slot=1,
      body="Family medicine physician for annual physicals, diabetes management, cold and flu, preventive care and chronic conditions."),
 dict(id="p8", type="provider", title="Dr. Laura Bennett", specialty="Neurology", location="Springfield-Demo", lon=-84.17, lat=39.69, accepting="yes", next_slot=9,
      body="Neurologist for migraines, headaches, epilepsy, memory loss, dizziness, numbness and stroke follow-up."),
 dict(id="p9", type="provider", title="Dr. James Whitfield", specialty="Orthopedic Surgery", location="Centerville", lon=-84.16, lat=39.63, accepting="yes", next_slot=4,
      body="Orthopedic surgeon for shoulder, rotator cuff, back pain, spine and fracture care."),
 dict(id="p10", type="provider", title="Dr. Naomi Epstein", specialty="Oncology", location="Dayton", lon=-84.19, lat=39.76, accepting="yes", next_slot=2,
      body="Medical oncologist treating breast cancer, lung cancer, chemotherapy, immunotherapy and survivorship."),
 # services
 dict(id="s1", type="service", title="Urgent Care", specialty="", location="Beavercreek", lon=-84.06, lat=39.71, accepting="yes", next_slot=0,
      body="Walk-in urgent care for minor injuries, sprains, cuts, fever, sore throat, flu symptoms and X-rays. Open evenings and weekends."),
 dict(id="s2", type="service", title="Emergency Department", specialty="", location="Dayton", lon=-84.19, lat=39.76, accepting="yes", next_slot=0,
      body="24/7 emergency care for chest pain, stroke symptoms, severe injury, difficulty breathing and trauma."),
 dict(id="s3", type="service", title="Women's Imaging & Mammography", specialty="", location="Centerville", lon=-84.16, lat=39.63, accepting="yes", next_slot=3,
      body="3D mammography, breast ultrasound, bone density scans and women's imaging with same-week appointments."),
 dict(id="s4", type="service", title="Physical Therapy & Sports Medicine", specialty="", location="Springboro", lon=-84.23, lat=39.56, accepting="yes", next_slot=1,
      body="Physical therapy after surgery, sports rehab, back and knee pain, balance and mobility programs."),
 dict(id="s5", type="service", title="Heart & Vascular Institute", specialty="", location="Springfield-Demo", lon=-84.17, lat=39.69, accepting="yes", next_slot=2,
      body="Comprehensive cardiac care including stress tests, echocardiograms, cardiac rehab, heart attack and arrhythmia treatment."),
 dict(id="s6", type="service", title="Telehealth Video Visit", specialty="", location="Online", lon=-84.19, lat=39.76, accepting="yes", next_slot=0,
      body="Same-day virtual visits for cold, flu, rashes, pink eye, UTI, prescription refills and mental health check-ins."),
 dict(id="s7", type="service", title="Sleep Center", specialty="", location="Beavercreek", lon=-84.06, lat=39.71, accepting="yes", next_slot=14,
      body="Sleep studies for sleep apnea, insomnia, snoring, restless legs and daytime fatigue."),
 # conditions
 dict(id="c1", type="condition", title="Knee Pain", specialty="Orthopedics", location="", lon=-84.19, lat=39.76, accepting="", next_slot=-1,
      body="Knee pain can come from arthritis, meniscus tears, ligament injuries or overuse. Treatment ranges from physical therapy to knee replacement."),
 dict(id="c2", type="condition", title="Heart Attack Symptoms", specialty="Cardiology", location="", lon=-84.19, lat=39.76, accepting="", next_slot=-1,
      body="Chest pressure, shortness of breath, nausea, pain radiating to the arm or jaw. Call 911 immediately."),
 dict(id="c3", type="condition", title="Type 2 Diabetes", specialty="Primary Care", location="", lon=-84.19, lat=39.76, accepting="", next_slot=-1,
      body="Increased thirst, frequent urination, fatigue and blurred vision. Managed with diet, exercise, medication and monitoring blood sugar."),
 dict(id="c4", type="condition", title="Migraine", specialty="Neurology", location="", lon=-84.19, lat=39.76, accepting="", next_slot=-1,
      body="Recurring severe headaches often with nausea and sensitivity to light and sound. Treatments include preventive medication and lifestyle changes."),
 dict(id="c5", type="condition", title="Sleep Apnea", specialty="Sleep Medicine", location="", lon=-84.19, lat=39.76, accepting="", next_slot=-1,
      body="Breathing repeatedly stops during sleep causing loud snoring, gasping and daytime tiredness. Diagnosed with a sleep study."),
 dict(id="c6", type="condition", title="Eczema", specialty="Dermatology", location="", lon=-84.19, lat=39.76, accepting="", next_slot=-1,
      body="Dry, itchy, inflamed skin. Managed with moisturizers, topical steroids and trigger avoidance."),
 # locations
 dict(id="l1", type="location", title="Springfield-Demo Medical Center", specialty="", location="Springfield-Demo", lon=-84.17, lat=39.69, accepting="", next_slot=-1,
      body="Main hospital campus: emergency, heart and vascular, neurology, surgery, labs and imaging. Free parking and valet."),
 dict(id="l2", type="location", title="Beavercreek Health Pavilion", specialty="", location="Beavercreek", lon=-84.06, lat=39.71, accepting="", next_slot=-1,
      body="Outpatient pavilion with urgent care, pediatrics, dermatology and sleep center. Evening and weekend hours."),
 dict(id="l3", type="location", title="Springboro Family Health", specialty="", location="Springboro", lon=-84.23, lat=39.56, accepting="", next_slot=-1,
      body="Primary care, physical therapy and sports medicine in a convenient suburban location."),
]


def create_index():
    try:
        r.execute_command("FT.DROPINDEX", IDX, "DD")
    except redis.ResponseError:
        pass
    r.execute_command(
        "FT.CREATE", IDX, "ON", "HASH", "PREFIX", "1", "care:", "SCHEMA",
        "title", "TEXT", "WEIGHT", "3.0",
        "body", "TEXT",
        "specialty", "TEXT", "WEIGHT", "2.0",
        "type", "TAG",
        "location", "TAG",
        "accepting", "TAG",
        "next_slot", "NUMERIC", "SORTABLE",
        "geo", "GEO",
        "vec", "VECTOR", "HNSW", "6", "TYPE", "FLOAT32", "DIM", str(DIM), "DISTANCE_METRIC", "COSINE",
    )


def load():
    create_index()
    pipe = r.pipeline()
    for d in DOCS:
        text = f"{d['title']}. {d['specialty']}. {d['body']}"
        pipe.hset(f"care:{d['id']}", mapping={
            "title": d["title"], "body": d["body"], "specialty": d["specialty"],
            "type": d["type"], "location": d["location"].replace("-", "_") or "NA",
            "accepting": d["accepting"] or "NA", "next_slot": d["next_slot"],
            "geo": f"{d['lon']},{d['lat']}", "vec": embed(text),
        })
    pipe.execute()
    print(f"Loaded {len(DOCS)} docs into {IDX}")


# ---------------------------------------------------------------- query helpers
def esc(q):  # strip characters that are special in the query syntax
    return "".join(c if c.isalnum() or c.isspace() else " " for c in q)


def timed(fn):
    t = time.perf_counter()
    res = fn()
    return res, (time.perf_counter() - t) * 1000


def parse(res):
    out = []
    for i in range(1, len(res), 2):
        f = res[i + 1]
        d = {f[j].decode(): f[j + 1].decode(errors="ignore") for j in range(0, len(f), 2) if f[j].decode() != "vec"}
        out.append(d)
    return out


def show(label, rows, ms, cols=("title", "type")):
    print(f"\n  [{label}]  {ms:.1f} ms client round-trip, {len(rows)} results")
    for d in rows[:5]:
        print("   -", " | ".join(str(d.get(c, "")) for c in cols))
    if not rows:
        print("   - (no results)  <-- the dead end")


def keyword(q, extra="", n=5):
    words = esc(q).split()
    qs = " ".join(words) + (" " + extra if extra else "")
    res = r.execute_command("FT.SEARCH", IDX, qs, "RETURN", "3", "title", "type", "specialty", "LIMIT", "0", str(n))
    return parse(res)


def fuzzy(q, extra="", n=5):
    # %%word%% = Levenshtein distance up to 2 -> handles misspellings
    words = [f"%%{w}%%" for w in esc(q).split() if len(w) > 3]
    qs = "(" + "|".join(words) + ")" + (" " + extra if extra else "")
    res = r.execute_command("FT.SEARCH", IDX, qs, "RETURN", "3", "title", "type", "specialty", "LIMIT", "0", str(n))
    return parse(res)


def vector(q, prefilter="*", n=5):
    res = r.execute_command(
        "FT.SEARCH", IDX, f"({prefilter})=>[KNN {n} @vec $v AS dist]",
        "PARAMS", "2", "v", embed(q), "SORTBY", "dist", "RETURN", "4", "title", "type", "specialty", "dist",
        "DIALECT", "2")
    return parse(res)


def hybrid(q, filt=None, n=5):
    args = ["FT.HYBRID", IDX, "SEARCH", esc(q), "YIELD_SCORE_AS", "text_score",
            "VSIM", "@vec", "$v", "KNN", "2", "K", "20", "YIELD_SCORE_AS", "vec_score"]
    if filt:
        args += ["FILTER", filt]
    args += ["COMBINE", "RRF", "2", "WINDOW", "20", "LOAD", "3", "@title", "@type", "@specialty",
             "LIMIT", "0", str(n), "PARAMS", "2", "v", embed(q)]
    try:
        res = r.execute_command(*args)
        return parse_hybrid(res)
    except redis.ResponseError as e:
        print("   (FT.HYBRID unavailable on this server:", e, ") -> using KNN+prefilter fallback")
        return vector(q, filt or "*", n)


def parse_hybrid(res):
    # RESP3/2 shapes differ by client version; handle the flat-array form defensively
    rows = []
    if isinstance(res, list):
        for item in res:
            if isinstance(item, list) and item and isinstance(item[0], (bytes, str)):
                try:
                    rows.append({(k.decode() if isinstance(k, bytes) else k): (v.decode(errors="ignore") if isinstance(v, bytes) else v)
                                 for k, v in zip(item[::2], item[1::2])})
                except Exception:
                    pass
        if not rows and len(res) > 3 and isinstance(res[3], list):
            for item in res[3]:
                if isinstance(item, list):
                    rows.append({(k.decode() if isinstance(k, bytes) else k): (v.decode(errors="ignore") if isinstance(v, bytes) else v)
                                 for k, v in zip(item[::2], item[1::2])})
    return rows


def compare(q, filt=None):
    print(f"\n=== Query: \"{q}\" ===")
    rows, ms = timed(lambda: keyword(q));   show("1 keyword / BM25 (what OpenSearch does by default)", rows, ms)
    rows, ms = timed(lambda: fuzzy(q));     show("2 fuzzy keyword (typo-tolerant)", rows, ms)
    rows, ms = timed(lambda: vector(q));    show("3 vector / semantic", rows, ms)
    rows, ms = timed(lambda: hybrid(q, filt)); show("4 HYBRID (BM25 + vector, RRF)", rows, ms)


# ---------------------------------------------------------------- guided script
STEPS = [
 ("A. Exact name / intent: 'schedule Dr. Peters' -> provider + availability",
  lambda: (compare("Dr Madeline Peters"),
           print("\n  Structured: accepting new patients, soonest slot, orthopedics:"),
           [show("filter+sort", parse(r.execute_command("FT.SEARCH", IDX,
               "@type:{provider} @accepting:{yes} @specialty:orthopedic", "SORTBY", "next_slot", "ASC",
               "RETURN", "3", "title", "specialty", "next_slot", "LIMIT", "0", "5")), 0, cols=("title", "specialty", "next_slot"))])),
 ("B. Misspelled: 'cardeologist near springfield'",
  lambda: compare("cardeologist near springfield")),
 ("C. Conversational / long-tail: 'my knee hurts when I climb stairs'",
  lambda: compare("my knee hurts when I climb stairs")),
 ("D. No shared vocabulary: 'I can't sleep and I'm tired all day'",
  lambda: compare("I can't sleep and I'm tired all day")),
 ("E. Hybrid with business filters: 'rash that won't go away' at Beavercreek, accepting new patients",
  lambda: compare("rash that won't go away", "@location:{Beavercreek} @accepting:{yes}")),
 ("F. Geo: care within 8 km of a patient in Centerville",
  lambda: show("geo filter", parse(r.execute_command("FT.SEARCH", IDX,
      "@geo:[-84.16 39.63 8 km] @type:{service|provider}", "RETURN", "2", "title", "location", "LIMIT", "0", "10")), 0, cols=("title", "location"))),
]


def run():
    for title, fn in STEPS:
        print("\n" + "#" * 78 + f"\n# {title}\n" + "#" * 78)
        fn()
        input("\n  <enter> for next...")


def availability():
    print("Dr. Peters next_slot BEFORE:", r.hget("care:p1", "next_slot"))
    r.hset("care:p1", "next_slot", 0)  # e.g. cancellation pushed from Epic scheduling feed
    rows = parse(r.execute_command("FT.SEARCH", IDX, "@type:{provider} @specialty:orthopedic", "SORTBY", "next_slot", "ASC",
                                   "RETURN", "2", "title", "next_slot"))
    print("Index is updated synchronously - same query immediately:", rows[:2])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "load": load()
    elif cmd == "run": run()
    elif cmd == "q": compare(" ".join(sys.argv[2:]))
    elif cmd == "availability": availability()
    else: print(__doc__)
