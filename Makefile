test:
	uv run pytest -v 2>&1 | tee logs/pytest.log

test-unit:
	uv run pytest -v -m "not integration" 2>&1 | tee logs/pytest.log

test-quality:
	uv run pytest tests/test_agent_quality.py -v 2>&1 | tee logs/pytest.log
