# xenRAG

## Setup

```bash
git clone https://github.com/xen-devs/xenRAG.git
cd xenRAG

# Install dependencies
uv sync
```

### Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

## Start Services

```bash
docker compose up -d
```

## Ingest Data

```bash
uv run python scripts/ingest.py data/Electronics_B075X8471B_B00ZV9RDKK_reviews.jsonl 
```

### For limitted ingestion
```bash
uv run python scripts/ingest.py data/Electronics_B075X8471B_B00ZV9RDKK_reviews.jsonl --limit 500
```


## Run

```bash
uv run python cli.py
```