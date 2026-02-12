.PHONY: lint format fix check complexity lint-all format-all fix-all check-all complexity-all precommit-install

CORE_DIR := steamlensai/core
RUFF := uv tool run ruff
PRE_COMMIT := uv tool run pre-commit

lint:
	$(RUFF) check $(CORE_DIR) --ignore C901

format:
	$(RUFF) format $(CORE_DIR)

fix:
	$(RUFF) check $(CORE_DIR) --fix --unsafe-fixes --ignore C901
	$(RUFF) format $(CORE_DIR)

check: lint
	@echo "format check"
	$(RUFF) format $(CORE_DIR) --check

complexity:
	$(RUFF) check $(CORE_DIR) --select C901

lint-all:
	$(RUFF) check . --ignore C901

format-all:
	$(RUFF) format .

fix-all:
	$(RUFF) check . --fix --unsafe-fixes --ignore C901
	$(RUFF) format .

check-all: lint-all
	@echo "format check (all)"
	$(RUFF) format . --check

complexity-all:
	$(RUFF) check . --select C901

precommit-install:
	$(PRE_COMMIT) install
