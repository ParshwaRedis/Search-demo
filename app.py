#!/usr/bin/env python3
"""
Web UI for the Acme Health healthcare search demo (Redis Query Engine).

  source .venv/bin/activate && pip install flask
  export REDIS_URL="redis://default:<password>@<host>:<port>"
  python demo.py load      # once: index + synthetic data
  python extras.py         # once: autocomplete, synonyms, intent router
  python app.py            # open http://localhost:5050
"""
import time, statistics
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import redis
from flask import Flask, jsonify, request, send_from_directory
import demo, extras

app = Flask(__name__, static_folder="static")
r, IDX = demo.r, demo.IDX
FIELDS = ["title", "type", "specialty", "location", "next_slot", "body", "accepting"]
TYPES = {}
for d in demo.DOCS:
    TYPES[d["title"].lower()] = d["type"]; TYPES[d["title"].replace("Dr. ", "").lower()] = d["type"]
    if d["specialty"]: TYPES.setdefault(d["specialty"].lower(), "specialty")
import os
GAP = float(os.environ.get("GAP", 0.12))  # patient view: drop semantic-only hits this far behind the best match
GENERIC = {"schedule", "book", "appointment", "appt", "dr", "doctor", "with", "see", "make", "an", "the", "for",
           "me", "my", "next", "available", "visit", "want", "need", "to", "can", "please", "find"}


def kw_titles(M):
    return {x["title"] for m in ("keyword", "fuzzy") for x in M[m]["rows"]}


STATS = {"searches": 0, "kw_dead": 0, "hy_answered": 0, "no_answer": 0, "intents": {}, "log": []}


def dec(x):
    return x.decode(errors="ignore") if isinstance(x, bytes) else x


def ms_since(t):
    return round((time.perf_counter() - t) * 1000, 2)


def filt_str(location, accepting):
    p = []
    if location and location != "any": p.append("@location:{%s}" % location.replace("-", "_"))
    if accepting: p.append("@accepting:{yes}")
    return " ".join(p)


def clean(rows):
    for d in rows:
        loc = d.get("location", "")
        d["location"] = "" if loc == "NA" else loc.replace("_", "-")
        if d.get("accepting") == "NA": d["accepting"] = ""
        try: d["next_slot"] = int(float(d.get("next_slot", -1)))
        except ValueError: d["next_slot"] = -1
        if "dist" in d: d["dist"] = round(float(d["dist"]), 3)
    return rows


STOP = {"i", "im", "ive", "my", "me", "and", "or", "the", "a", "an", "is", "it", "its", "of", "to", "in", "on", "at",
        "for", "when", "what", "how", "can", "cant", "t", "s", "all", "that", "with", "near", "best", "won", "wont",
        "get", "go", "up", "am", "be", "do", "dont", "this", "there", "where", "who"}


def text_query(q, f=""):
    """Text half of hybrid: any meaningful word (OR), plus fuzzy variants for longer words, plus filters."""
    parts = []
    for w in demo.esc(q).split():
        lw = w.lower()
        if len(lw) < 3 or lw in STOP or lw in GENERIC:
            continue
        parts.append(f"{lw}|%%{lw}%%" if len(lw) >= 5 else lw)
    base = "(" + "|".join(parts) + ")" if parts else "@type:{__none__}"   # nothing meaningful -> text side empty
    return base + (" " + f if f else "")


def hybrid_args(q, vec, f, n=5, load=True):
    a = ["FT.HYBRID", IDX, "SEARCH", text_query(q, f), "VSIM", "@vec", "$v", "KNN", "2", "K", "20"]
    if f: a += ["FILTER", f]
    a += ["COMBINE", "RRF", "2", "WINDOW", "20"]
    if load: a += ["LOAD", str(len(FIELDS))] + ["@" + x for x in FIELDS]
    return a + ["LIMIT", "0", str(n), "PARAMS", "2", "v", vec]


def ping_rtt(n=15):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); r.ping(); ts.append((time.perf_counter() - t) * 1000)
    return round(statistics.median(ts), 2)


def spellcheck(q):
    try:
        res = r.execute_command("FT.SPELLCHECK", IDX, demo.esc(q), "DISTANCE", "2")
    except redis.ResponseError:
        return None
    fixes = {}
    items = res.items() if isinstance(res, dict) else [(dec(i[1]), i[2]) for i in (res or [])
                                                        if isinstance(i, (list, tuple)) and len(i) >= 3]
    for term, sugs in items:
        term = dec(term)
        if isinstance(sugs, dict): sugs = [[v, k] for k, v in sugs.items()]
        cands = [(float(dec(s[0])), dec(s[1])) for s in (sugs or []) if isinstance(s, (list, tuple)) and len(s) >= 2]
        if cands:
            fixes[term.lower()] = max(cands)[1]
    if not fixes: return None
    return {"fixes": fixes, "corrected": " ".join(fixes.get(w.lower(), w) for w in demo.esc(q).split())}


# ---------------------------------------------------------------- routes
@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/info")
def info():
    ver = dec(r.info("server").get("redis_version", "?"))
    try:
        n = r.execute_command("FT.INFO", IDX)
        d = {dec(n[i]): n[i + 1] for i in range(0, len(n) - 1, 2)}
        docs = int(dec(d.get("num_docs", 0)))
    except Exception:
        docs = 0
    return jsonify(version=ver, docs=docs, rtt=ping_rtt())


@app.get("/api/suggest")
def suggest():
    p = (request.args.get("p") or "").strip()
    if not p: return jsonify(items=[], ms=0)
    args = ["FT.SUGGET", extras.SUG, p, "MAX", "6"] + (["FUZZY"] if len(p) >= 4 else [])
    t = time.perf_counter()
    res = r.execute_command(*args) or []
    ms = ms_since(t)
    items = [{"text": dec(s), "type": TYPES.get(dec(s).lower(), "search")} for s in res]
    return jsonify(items=items, ms=ms)


def run_mode(name, fn):
    t = time.perf_counter(); note = None
    try:
        rows, note = fn()
        rows, err = clean(rows), None
    except redis.ResponseError as e:
        rows, err = [], str(e)
    return {"mode": name, "rows": rows[:5], "ms": ms_since(t), "error": err, "note": note}


@app.post("/api/search")
def search():
    b = request.get_json(force=True)
    q = (b.get("q") or "").strip()
    if not q: return jsonify(modes=[])
    f = filt_str(b.get("location", "any"), bool(b.get("accepting")))

    t = time.perf_counter(); vec = demo.embed(q); embed_ms = ms_since(t)
    t = time.perf_counter(); intent = extras.classify(q, vec); intent["ms"] = ms_since(t)
    words = demo.esc(q).split()
    ret = ["RETURN", str(len(FIELDS))] + FIELDS

    def keyword():
        qs = " ".join(words) + (" " + f if f else "")
        return demo.parse(r.execute_command("FT.SEARCH", IDX, qs, *ret, "LIMIT", "0", "5")), None

    def fuzzy():
        ws = [f"%%{w}%%" for w in words if len(w) > 3]
        if not ws: return [], "no words long enough for fuzzy"
        qs = "(" + "|".join(ws) + ")" + (" " + f if f else "")
        return demo.parse(r.execute_command("FT.SEARCH", IDX, qs, *ret, "LIMIT", "0", "5")), None

    def vector():
        res = r.execute_command("FT.SEARCH", IDX, f"({f or '*'})=>[KNN 5 @vec $v AS dist]", "PARAMS", "2", "v", vec,
                                "SORTBY", "dist", "RETURN", str(len(FIELDS) + 1), *FIELDS, "dist", "DIALECT", "2")
        return demo.parse(res), None

    def hybrid():
        try:
            rows = demo.parse_hybrid(r.execute_command(*hybrid_args(q, vec, f)))
            txt = {x.get("title") for x in demo.parse(r.execute_command(
                "FT.SEARCH", IDX, text_query(q, f), "RETURN", "1", "title", "LIMIT", "0", "50"))}
            for x in rows:
                x["via"] = "words + meaning" if x.get("title") in txt else "meaning"
            return rows, None
        except redis.ResponseError as e:
            rows, _ = vector()
            return rows, f"FT.HYBRID failed ({e}); showing KNN+prefilter fallback"

    modes = [run_mode("keyword", keyword), run_mode("fuzzy", fuzzy),
             run_mode("vector", vector), run_mode("hybrid", hybrid)]
    M = {m["mode"]: m for m in modes}

    best = M["vector"]["rows"][0]["dist"] if M["vector"]["rows"] else 1.0
    no_answer = (not M["keyword"]["rows"]) and best > extras.NOANSWER_T and intent["route"] != "emergency"
    spell = spellcheck(q) if not M["keyword"]["rows"] and not no_answer else None

    emergency = None
    if intent["route"] == "emergency":
        ed = demo.parse(r.execute_command("FT.SEARCH", IDX, "@type:{service} @title:emergency", *ret, "LIMIT", "0", "1"))
        emergency = clean(ed)[0] if ed else None

    # --- patient-facing relevance: entity resolution + cutoff (Compare view keeps raw lists) ---
    entity, alternates = None, []
    names = [w for w in words if w.lower() not in GENERIC and len(w) > 2]
    if names and intent["route"] != "emergency":
        hits = clean(demo.parse(r.execute_command("FT.SEARCH", IDX, "@type:{provider} @title:(%s)" % "|".join(names),
                                                  *ret, "LIMIT", "0", "1")))
        if hits:
            entity = hits[0]
            spec = demo.esc(entity.get("specialty", ""))
            if spec:
                alt = clean(demo.parse(r.execute_command(
                    "FT.SEARCH", IDX, "@type:{provider} @accepting:{yes} @specialty:(%s)" % spec,
                    *ret, "SORTBY", "next_slot", "ASC", "LIMIT", "0", "4")))
                alternates = [a for a in alt if a["title"] != entity["title"]][:3]
    vd = {x["title"]: x.get("dist", 1) for x in M["vector"]["rows"]}
    patient_rows = [x for x in M["hybrid"]["rows"] if x["title"] in kw_titles(M) or vd.get(x["title"], best) <= best + GAP]

    S = STATS; S["searches"] += 1
    kw_empty = not M["keyword"]["rows"]
    S["kw_dead"] += int(kw_empty); S["no_answer"] += int(no_answer)
    S["hy_answered"] += int(kw_empty and bool(M["hybrid"]["rows"]) and not no_answer)
    S["intents"][intent["route"]] = S["intents"].get(intent["route"], 0) + 1
    S["log"].insert(0, {"q": q, "kw": len(M["keyword"]["rows"]), "hy": len(M["hybrid"]["rows"]),
                        "intent": intent["route"], "no_answer": no_answer}); S["log"] = S["log"][:8]

    return jsonify(modes=modes, embed_ms=embed_ms, intent=intent, best_dist=best, no_answer=no_answer,
                   noanswer_t=extras.NOANSWER_T, spell=spell, emergency=emergency, stats=S,
                   entity=entity, alternates=alternates, patient_rows=patient_rows, gap=GAP)


@app.post("/api/availability")
def availability():
    b = request.get_json(force=True)
    r.hset("care:p1", "next_slot", int(b.get("slot", 0)))
    return jsonify(ok=True)


@app.post("/api/loadtest")
def loadtest():
    b = request.get_json(force=True)
    q = b.get("q") or "my knee hurts when I climb stairs"
    n, c = min(int(b.get("n", 1000)), 5000), max(1, min(int(b.get("concurrency", 20)), 100))
    vec = demo.embed(q)                      # embed once: we are load-testing Redis, not the model
    args = hybrid_args(q, vec, "", n=10, load=False)
    client = redis.Redis.from_url(demo.URL, max_connections=c + 5)
    try:
        client.execute_command(*args)
    except redis.ResponseError:              # no FT.HYBRID on this server -> KNN
        args = ["FT.SEARCH", IDX, "*=>[KNN 10 @vec $v AS dist]", "PARAMS", "2", "v", vec, "NOCONTENT", "DIALECT", "2"]
    lat, errors = [], 0

    def one(_):
        t = time.perf_counter()
        try:
            client.execute_command(*args); return (time.perf_counter() - t) * 1000
        except Exception:
            return None

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=c) as ex:
        for v in ex.map(one, range(n)):
            if v is None: errors += 1
            else: lat.append(v)
    wall = time.perf_counter() - t0
    client.close()
    if not lat: return jsonify(error="all queries failed", errors=errors)
    a = np.array(lat)
    hi = float(np.percentile(a, 99)) * 1.15
    hist, edges = np.histogram(np.clip(a, 0, hi), bins=24, range=(0, hi))
    return jsonify(q=q, n=n, concurrency=c, errors=errors, qps=round(len(lat) / wall),
                   p50=round(float(np.percentile(a, 50)), 2), p95=round(float(np.percentile(a, 95)), 2),
                   p99=round(float(np.percentile(a, 99)), 2), rtt=ping_rtt(),
                   hist=hist.tolist(), edges=[round(float(e), 2) for e in edges], cmd=args[0])


if __name__ == "__main__":
    r.hset("care:p1", "next_slot", 1)   # reset demo state: Dr. Peters back to "Tomorrow"
    print("Reset Dr. Peters availability to Tomorrow")
    print("Warming embedding model..."); demo.embed("warmup")
    app.run(host="127.0.0.1", port=5050, debug=False, threaded=True)
