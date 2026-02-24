# Contributing

Guidelines for contributing to Yipoodle.

---

## Development Setup

```bash
# Clone and set up
git clone https://github.com/Ela-El-maker/yipoodle.git
cd yipoodle
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Verify setup
python -m src.cli --help
pytest -q
```

---

## Development Workflow

1. **Create a feature branch** from `main`:

   ```bash
   git checkout -b feature/my-feature main
   ```

2. **Make changes** following the code conventions below.

3. **Write tests** for any new functionality — see [testing.md](testing.md).

4. **Run the full test suite** before committing:

   ```bash
   pytest -q
   ```

   All 254+ tests must pass.

5. **Commit** with a clear, descriptive message:

   ```bash
   git commit -m "Add hybrid retrieval alpha tuning for live sources"
   ```

6. **Push and open a pull request** against `main`.

---

## Code Conventions

### Python Style

- **Python 3.11+** with `from __future__ import annotations`.
- **Type hints** on all function signatures.
- **Docstrings** on public functions and classes.
- No unused imports or variables.
- Prefer explicit over implicit.

### Module Organization

- **CLI commands** go in `src/cli.py` as `cmd_<name>()` functions.
- **Business logic** goes in `src/apps/<module>.py` — CLI functions should be thin wrappers.
- **Data models** go in `src/core/schemas.py` as Pydantic models.
- **Validation logic** goes in `src/core/validation.py` or `semantic_validation.py`.
- **Configuration** goes in YAML files under `config/`.

### Naming

- Functions: `snake_case` — e.g., `build_index()`, `run_research()`.
- Classes: `PascalCase` — e.g., `SimpleBM25Index`, `MiniGPT`.
- Constants: `UPPER_SNAKE_CASE` — e.g., `MAX_RESULTS`.
- CLI subcommands: `kebab-case` — e.g., `build-index`, `sync-papers`.
- Config keys: `snake_case` — e.g., `max_results`, `embedding_model`.

### File Naming

- Source modules: `snake_case.py` — e.g., `paper_search.py`, `kb_store.py`.
- Test files: `test_<module_name>.py` — mirrors the source module.
- Config files: `snake_case.yaml` — e.g., `sources.yaml`, `automation.yaml`.
- Domain configs: `sources_<domain>.yaml` — e.g., `sources_nlp.yaml`.

---

## Adding a New CLI Command

1. Add a parser in `src/cli.py`:

   ```python
   p = sub.add_parser("my-command", help="description")
   p.add_argument("--arg", default="value")
   p.set_defaults(func=cmd_my_command)
   ```

2. Add the handler function:

   ```python
   def cmd_my_command(args):
       from src.apps.my_module import my_function
       result = my_function(arg=args.arg)
       print(json.dumps(result, indent=2))
   ```

3. Add tests in `tests/test_my_module.py`.

4. Update [docs/cli-reference.md](cli-reference.md).

---

## Adding a New Source Connector

1. Add a `search_<source>()` function in `src/apps/paper_search.py`:

   ```python
   def search_my_source(query: str, max_results: int = 10) -> list[PaperRecord]:
       resp = _request_get_with_retry(endpoint, params={...})
       # Parse response into PaperRecord list
       return papers
   ```

2. Register it in `src/apps/paper_sync.py`'s source dispatch.

3. Add default config in `config/sources.yaml`:

   ```yaml
   sources:
     my_source:
       enabled: false
       endpoint: "https://api.example.com/search"
       max_results: 10
   ```

4. Add tests with mocked HTTP responses.

---

## Adding a New Domain Config

Use the scaffold command:

```bash
python -m src.cli scaffold-domain-config --domain "my domain" \
  --out config/domains/sources_my_domain.yaml
```

Or manually create a YAML file following the structure in `config/sources.yaml`.

---

## Testing Requirements

- All new code must have corresponding tests.
- Tests must work fully offline — mock all HTTP calls.
- Use `tmp_path` for any file I/O.
- Run `pytest -q` and ensure all tests pass before submitting.
- Target: zero regressions on the existing 254 tests.

---

## Documentation

When making significant changes:

1. Update the relevant docs in `docs/`.
2. If adding CLI commands, update [docs/cli-reference.md](cli-reference.md).
3. If modifying config, update [docs/configuration.md](configuration.md).
4. If adding features, update [docs/system-overview.md](system-overview.md) and the README.

---

## Commit Message Guidelines

Format: `<type>: <description>`

Types:

- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code restructuring without behavior change
- `docs`: Documentation only
- `test`: Adding or modifying tests
- `config`: Configuration file changes
- `build`: Build system or dependency changes

Examples:

```
feat: add ORCID source connector with retry logic
fix: handle empty extraction in corpus health gate
docs: update CLI reference with new monitor commands
test: add hybrid retrieval alpha boundary tests
refactor: extract cache key logic into dedicated function
```
