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
cp .env.example .env
# Edit .env with your settings
```

## Start Services

```bash
docker compose up -d
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