# Testing Guide

This document describes the test suite and how to run tests.

## Test Structure

```
tests/
 conftest.py               # Pytest fixtures
 test_parser.py            # Parser tests (UNIT-parse-01/02, UNIT-yaml-01)
 test_slug_generator.py    # Slug generation tests (UNIT-slug-01)
 test_apf_linter.py        # APF validation tests (UNIT-apf-a, UNIT-tag-01, LINT-cloze)
 test_state_db.py          # Database tests
 integration/
     test_anki_client.py   # AnkiConnect tests (INT-01, INT-crud-01)
     test_sync_determinism.py  # Sync flow tests (INT-02, REGR-det-01)
```

## Running Tests

### Install Test Dependencies

**Using uv (Recommended):**

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup environment with all dependencies
uv sync --all-extras

# Activate virtual environment
source .venv/bin/activate
```

**Using pip:**

```bash
pip install -e ".[dev]"
```

### Run All Tests

```bash
pytest
```

Or with uv directly (without activating venv):

```bash
uv run pytest
```

### Run with Coverage

```bash
pytest --cov=src/obsidian_anki_sync --cov-report=html
```

View coverage report in `htmlcov/index.html`.

### Run Specific Test Files

```bash
# Parser tests
pytest tests/test_parser.py

# APF linter tests
pytest tests/test_apf_linter.py

# Integration tests
pytest tests/integration/
```

### Run Tests by Marker

```bash
# Run only unit tests
pytest -m unit

# Skip slow tests
pytest -m "not slow"
```

## Test Coverage

### Unit Tests

**Parser Tests (test_parser.py)**
- UNIT-yaml-01: YAML frontmatter validation
  - Valid frontmatter parsing
  - Missing frontmatter error
  - Missing required fields error
  - Invalid YAML syntax error
- UNIT-parse-01: Single Q/A pair parsing
  - Basic Q/A extraction
  - Follow-ups and references
- UNIT-parse-02: Multi-pair Q/A parsing
  - Multiple Q/A blocks
  - Card index assignment
- File discovery
  - q-*.md filtering
  - Recursive scanning

**Slug Generator Tests (test_slug_generator.py)**
- UNIT-slug-01: Slug generation
  - Basic slug formation
  - Sanitization of special characters
  - Collision resolution with hash6
  - Language suffix
  - Card index zero-padding
  - Determinism

**APF Linter Tests (test_apf_linter.py)**
- UNIT-apf-a: APF structure validation
  - Valid card validation
  - Missing sentinels error
  - Invalid card header error
- UNIT-tag-01: Tag validation
  - Valid tag count (3-6)
  - Too few/many tags error
  - Snake_case format validation
  - Non-language tag requirement
- LINT-cloze: Cloze validation
  - Dense numbering (1..N)
  - Sparse numbering error
  - Missing cloze warning

**Database Tests (test_state_db.py)**
- Database initialization
- Insert card
- Update card
- Get by slug/GUID/source
- Delete card
- Unique constraints

### Integration Tests

**AnkiConnect Client Tests (test_anki_client.py)**
- INT-01: API communication
  - Successful invoke
  - Error response handling
  - HTTP error handling
- INT-crud-01: CRUD operations
  - Find notes
  - Get notes info
  - Add note
  - Update note fields
  - Delete notes

**Sync Tests (test_sync_determinism.py)**
- INT-02: Sync flow (stub only, needs full implementation)
- REGR-det-01: Determinism test (stub only)

### E2E Tests

**Missing (Deferred to Phase 9):**
- E2E-01: Process 5 notes with 2 multi-pair
- E2E-02: Bilingual card generation
- E2E-03: Full sync cycle
- E2E-dryrun-01: Dry-run validation
- E2E-idemp-01: Idempotency test

## Test Fixtures

### Available Fixtures (conftest.py)

- `temp_dir`: Temporary directory for test files
- `test_config`: Test configuration with temp paths
- `sample_metadata`: Sample NoteMetadata object
- `sample_qa_pair`: Sample QAPair object
- `sample_note_content`: Full Obsidian note content

### Example Usage

```python
def test_example(temp_dir, sample_metadata):
    """Test using fixtures."""
    test_file = temp_dir / "test.md"
    test_file.write_text("content")

    assert sample_metadata.id == "test-001"
```

## Writing New Tests

### Unit Test Template

```python
class TestFeature:
    """Test feature description."""

    def test_basic_case(self):
        """Test basic functionality."""
        result = function_under_test(input)
        assert result == expected

    def test_edge_case(self):
        """Test edge case."""
        with pytest.raises(ExpectedException):
            function_under_test(invalid_input)
```

### Integration Test Template

```python
import respx
import httpx

@respx.mock
def test_external_api(mock_url):
    """Test external API interaction."""
    respx.post(mock_url).mock(
        return_value=httpx.Response(200, json={"result": "ok"})
    )

    client = ApiClient(mock_url)
    result = client.call()

    assert result == "ok"
```

## Continuous Integration

Tests are designed to run in CI environments:

```yaml
# Example GitHub Actions workflow (using uv)
- name: Install uv
  run: curl -LsSf https://astral.sh/uv/install.sh | sh

- name: Setup Python and dependencies
  run: |
    uv sync --all-extras

- name: Run tests
  run: |
    uv run pytest --cov --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
```

**Alternative with pip:**

```yaml
- name: Run tests
  run: |
    pip install -e ".[dev]"
    pytest --cov --cov-report=xml
```

## Known Test Limitations

1. **LLM Tests**: APF generator tests would require mocking OpenRouter API
2. **E2E Tests**: Require sample vault creation and full Anki setup
3. **Sync Integration**: Full sync flow tests are stubbed pending vault setup
4. **Golden Tests**: APF output golden files not yet created

## Adding Golden Tests

To add golden test files:

1. Create `tests/golden/` directory
2. Add expected APF output files
3. Compare LLM output against golden files
4. Update goldens when CARDS_PROMPT changes

Example:
```python
def test_apf_output_matches_golden():
    """Compare generated APF against golden file."""
    golden = Path("tests/golden/simple_card.html").read_text()
    generated = generate_apf_card(...)
    assert normalize_whitespace(generated) == normalize_whitespace(golden)
```

## Test Maintenance

- Update tests when requirements change
- Keep fixtures in sync with models
- Document test IDs (UNIT-*, INT-*, E2E-*) for traceability
- Run full test suite before committing

## Performance Testing

For performance validation:

```bash
# Time test execution
pytest --durations=10

# Profile specific test
pytest tests/test_parser.py --profile
```

Target: Tests should complete in <30 seconds (excluding E2E).
