# Healthcare Search Demo (Redis Query Engine)

A small web app that shows how one Redis database can power a whole patient search journey, on a synthetic dataset of providers, services and locations:

- **Full-text and keyword search**, with provider and next-available slot on top
- **Typo tolerance** (fuzzy matching) and "Did you mean" spellcheck
- **Synonyms** (for example "kids doctor" finds pediatricians)
- **Vector search** on meaning ("my knee hurts when I climb stairs")
- **Hybrid search** (`FT.HYBRID`) fusing keyword and vector rankings
- **Intent routing** (schedule, urgent care, directions, emergency) using vector similarity
- **Grounding**: low-similarity queries return "no answer" instead of a guess
- **Typeahead** autocomplete
- **Real-time updates**: change a field in Redis and the next search reflects it
- **Load test** and **search analytics** panels

All data is synthetic. There is no PHI.

## Requirements

- Python 3.10+
- A Redis database with the Redis Query Engine (search and query), such as [Redis Cloud](https://redis.io/cloud/) or Redis Stack / Redis 8
- First run downloads a small sentence-transformers embedding model

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export REDIS_URL="redis://default:<password>@<host>:<port>"   # defaults to redis://localhost:6379
```

## Run

```bash
python demo.py load    # once: create the index and load synthetic data
python extras.py       # once: autocomplete, synonyms, intent router
python app.py          # open http://localhost:5050
```

Try the example searches in the UI: `schedule Dr. Peters`, `cardeologist`, `kids doctor`, `my knee hurts when I climb stairs`, `I can't sleep and I'm tired all day`, `chest pain and I can't breathe`, `best pizza near me`. Use **Compare search types** to see keyword, fuzzy, vector and hybrid results side by side.

## Files

| File | Purpose |
|---|---|
| `demo.py` | Index schema, synthetic data, and the search functions (CLI and library) |
| `extras.py` | Autocomplete suggestions, synonym groups, intent router |
| `app.py` | Flask API and UI backend |
| `static/index.html` | Front end |

## Notes

This is a demo, not production code: the dataset is tiny, so judge the behavior rather than the absolute latency numbers.
