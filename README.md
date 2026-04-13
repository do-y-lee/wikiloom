# wikiloom

WikiLoom turns raw documents into a persistent, compounding knowledge base. The LLM reads sources and writes structured wiki pages.

## Development

### Installation

```bash
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
```

The spaCy `en_core_web_sm` model is required by the linking engine and its tests. Without it, `tests/test_linker.py` will skip every test.

### Running tests

```bash
pytest
```
