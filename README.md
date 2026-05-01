# xenRAG

## Setup

```bash
git clone https://github.com/xen-devs/xenRAG.git
cd xenRAG

# Install dependencies
uv sync
npm install
```

### Configure Environment

```bash
cp apps/backend/.env.example apps/backend/.env
```

## Start Services

```bash
docker compose up -d
```

## Database Migrations

Apply existing migrations (run this after setup or pulling new changes):

```bash
cd apps/backend
uv run alembic -c api/db/alembic.ini upgrade head
```

Create a new migration when you change models during development:

```bash
cd apps/backend
uv run alembic -c api/db/alembic.ini revision --autogenerate -m "describe changes"

```

## Ingest Data

```bash
npm run ingest -- data/Electronics_B075X8471B_B00ZV9RDKK_reviews.jsonl
```

### Limited ingestion

```bash
npm run ingest -- data/Electronics_B075X8471B_B00ZV9RDKK_reviews.jsonl --limit 500
```

## Run

```bash
npm run ai:cli
```

## Quick Evaluation

Use the generated quick eval file:

```bash
uv run python scripts/evaluate_bot.py \
  --eval-file data/eval_cases_10.json \
  --output-dir eval_outputs
```

This writes:

- `eval_outputs/evaluation.log`
- `eval_outputs/eval_<timestamp>.json`
- `eval_outputs/eval_<timestamp>.csv`

### Run with Turborepo

```bash
npm run dev
```

```bash
npm run build
npm run lint
npm run typecheck
npm run test
```

