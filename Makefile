ENV_NAME = nowcastpnn

.PHONY: help setup clean

help:
	@echo "Available commands:"
	@echo "  setup   : Creates the conda environment and installs pre-commit hooks."
	@echo "  clean   : Removes the conda environment."

# This is the main setup command
setup:
	@echo ">>> Creating conda environment '$(ENV_NAME)' from environment.yaml..."
	@micromamba env create -f environment.yaml
	@echo "\n>>> Installing pre-commit hooks..."
	@micromamba run -n $(ENV_NAME) pre-commit install
	@echo "\n>>> Setup complete. Activate the environment with: micromamba activate $(ENV_NAME)"

clean:
	@echo ">>> Removing conda environment '$(ENV_NAME)'..."
	@micromamba env remove -n $(ENV_NAME) --yes
