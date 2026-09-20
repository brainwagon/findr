HOST ?= findr.local
USER ?= markv
REMOTE_DIR ?= /home/markv/findr
SSH := ssh $(USER)@$(HOST)
SSH_TTY := ssh -t $(USER)@$(HOST)
PYTHON ?= python3
VENV ?= venv

RSYNC_EXCLUDES := \
	--exclude 'venv/' \
	--exclude '__pycache__/' \
	--exclude '.git/' \
	--exclude '.opencode/' \
	--exclude '.agents/' \
	--exclude '.claude/' \
	--exclude '.pytest_cache/' \
	--exclude 'olive-solve/'

.PHONY: test sync deploy requirements restart status logs olive

test:
	$(PYTHON) -m unittest discover -s tests

olive:
	$(CURDIR)/$(VENV)/bin/pip install -q maturin
	cd olive-solve && $(CURDIR)/$(VENV)/bin/maturin build --release --interpreter $(CURDIR)/$(VENV)/bin/python
	$(CURDIR)/$(VENV)/bin/pip install --force-reinstall --no-deps $$(ls -t olive-solve/target/wheels/olive_solve-*.whl | head -1)

sync:
	rsync -az --delete $(RSYNC_EXCLUDES) ./ $(USER)@$(HOST):$(REMOTE_DIR)/

deploy: sync
	$(SSH_TTY) 'sudo systemctl restart findr.service'
	@echo "Deployed. http://$(HOST)/"

requirements:
	$(SSH) '$(REMOTE_DIR)/venv/bin/pip install -r $(REMOTE_DIR)/requirements.txt'

restart:
	$(SSH_TTY) 'sudo systemctl restart findr.service'

status:
	$(SSH) 'systemctl is-enabled findr.service; systemctl status findr.service --no-pager | head -20'

logs:
	$(SSH_TTY) 'sudo journalctl -u findr.service -n 50 -f'
