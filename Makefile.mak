# =========================
# Project settings
# =========================
PYTHON := python
MODULE := src
REPORTS := reports
MODELS := lgbm xgb ridge extratrees

.DEFAULT_GOAL := help

# =========================
# Help
# =========================
help:
	@echo ""
	@echo "House Prices ML Pipeline - Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make train model=<name> folds=<k>"
	@echo "  make train-all folds=<k>"
	@echo "  make predict-kaggle model=<name>"
	@echo "  make ensemble type=<blend_mean|blend_weighted|stack>"
	@echo "  make predict-prod model_id=<model/alias> input=<csv>"
	@echo "  make test"
	@echo "  make clean"
	@echo ""
	@echo "Examples:"
	@echo "  make train model=ridge folds=5"
	@echo "  make train-all folds=3"
	@echo "  make predict-kaggle model=lgbm"
	@echo "  make ensemble type=blend_weighted"
	@echo "  make predict-prod model_id=ridge/latest input=data/new.csv"
	@echo ""

# =========================
# Training
# =========================
train:
	@if [ -z "$(model)" ]; then \
		echo "❌ Please specify model=<name>"; exit 1; \
	fi
	$(PYTHON) -m $(MODULE).train --model $(model) --folds $(folds)

train-all:
	@for m in $(MODELS); do \
		echo "==== Training $$m ===="; \
		$(PYTHON) -m $(MODULE).train --model $$m --folds $(folds); \
	done

# =========================
# Kaggle prediction
# =========================
predict-kaggle:
	@if [ -z "$(model)" ]; then \
		echo "❌ Please specify model=<name>"; exit 1; \
	fi
	$(PYTHON) -m $(MODULE).predict kaggle --model $(model)

ensemble:
	@if [ -z "$(type)" ]; then \
		echo "❌ Please specify type=<blend_mean|blend_weighted|stack>"; exit 1; \
	fi
	$(PYTHON) -m $(MODULE).predict kaggle --ensemble $(type)

# =========================
# Production / Registry
# =========================
predict-prod:
	@if [ -z "$(model_id)" ]; then \
		echo "❌ Please specify model_id=<model/alias>"; exit 1; \
	fi
	$(PYTHON) -m $(MODULE).predict prod \
		--model-id $(model_id) \
		--input $(input)

# =========================
# Quality / hygiene
# =========================
test:
	pytest -q

clean:
	rm -rf $(REPORTS)/predictions
	rm -rf $(REPORTS)/*.csv
	rm -rf $(REPORTS)/*.npy
	rm -rf __pycache__ .pytest_cache
