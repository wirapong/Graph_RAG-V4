# Deployment

## 1. Streamlit Cloud

Recommended path: free, zero-ops.

### Steps
1. Push this repo to GitHub.
2. Go to <https://streamlit.io/cloud> → **New app**.
3. Repository: `<you>/graph_rag-V4`
   Main file: `streamlit_app.py`
   Python version: **3.11**
4. **Secrets** (paste in the dashboard):
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
   ```
5. Deploy. The app uses `requirements.txt` automatically.

### Notes
- The first run downloads `sentence-transformers/all-MiniLM-L6-v2` (~80 MB).
- If memory is tight, set `model_name=""` in `SemanticMetadataSearch` to force the hash-based fallback.
- `secrets.toml` is never committed (gitignored). The `.example` file is your template.

## 2. FastAPI on a server

```bash
# inside a venv
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=...
metadata-kg-api
# → 0.0.0.0:8000, OpenAPI docs at /docs
```

For production behind a reverse proxy use a proper ASGI runner:

```bash
uvicorn metadata_kg.api.routes:app \
    --host 0.0.0.0 --port 8000 \
    --workers 2 \
    --proxy-headers
```

## 3. Docker (optional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000 8501
CMD ["uvicorn", "metadata_kg.api.routes:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build:

```bash
docker build -t metadata-kg:0.1.0 .
docker run --rm -p 8000:8000 -e ANTHROPIC_API_KEY metadata-kg:0.1.0
```

## 4. GitHub Actions CI (suggested)

Create `.github/workflows/test.yml`:

```yaml
name: tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: pytest -v
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'metadata_kg'` | Run `pip install -e .` or `export PYTHONPATH=$PWD` |
| Sentence-transformers OOM | Fall back to hashed embeddings (no API change required — set offline) |
| LangChain not available log | Expected when `langchain` not installed; deterministic fallback works |
| HuggingFace download blocked | Set `HF_HOME` to a writable dir or pre-pack the model |
