# Folder struture
**TODO** - add a tree

# X-files
## ENV
[text](.env)

## Create and insert SQL from Python dict
[text](eu_sql.py)

## Load EUREKA dictionary from API
[text](eu_load_dict.py) 

## Consolidate files by group
[text](eu_consolidate_json.py)


# NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.

nvidia-smi
sudo apt-get update
sudo ubuntu-drivers autoinstall
sudo reboot



# **uv**
https://docs.astral.sh/uv/
https://docs.astral.sh/uv/getting-started/features/

cd eutest
    uv init
    uv venv --python 3.12
    source .venv/bin/activate or . .venv/bin/activate
    git init **TODO** - waiting for gitea
    touch .env **dotenv**
    mkdir my_dir
----------------------------------------------------
        uv add package name_pkg **it installs the package and adds it to the project**
        uv python install name_pkg **it installs the package but does not add it to the project**
-----------------------------------------------------
    deactivate


## Python versions - Installing and managing Python itself.
uv python install: Install Python versions.
uv python list: View available Python versions.
uv python find: Find an installed Python version.
uv python pin: Pin the current project to use a specific Python version.
uv python uninstall: Uninstall a Python version.

## Scripts. Executing standalone Python scripts, e.g., example.py.
uv run: Run a script.
uv add --script: Add a dependency to a script.
uv remove --script: Remove a dependency from a script.

## Projects. Creating and working on Python projects, i.e., with a pyproject.toml.
uv init: Create a new Python project.
uv add: Add a dependency to the project.
uv remove: Remove a dependency from the project.
uv sync: Sync the project's dependencies with the environment.
uv lock: Create a lockfile for the project's dependencies.
uv run: Run a command in the project environment.
uv tree: View the dependency tree for the project.
uv build: Build the project into distribution archives.
uv publish: Publish the project to a package index.

## Tools. Running and installing tools published to Python package indexes, e.g., ruff or black.
uvx / uv tool run: Run a tool in a temporary environment.
uv tool install: Install a tool user-wide.
uv tool uninstall: Uninstall a tool.
uv tool list: List installed tools.
uv tool update-shell: Update the shell to include tool executables.

## Creating virtual environments (replacing venv and virtualenv):
uv venv: Create a new virtual environment.

## Managing packages in an environment (replacing pip and pipdeptree):
uv pip install: Install packages into the current environment.
uv pip show: Show details about an installed package.
uv pip freeze: List installed packages and their versions.
uv pip check: Check that the current environment has compatible packages.
uv pip list: List installed packages.
uv pip uninstall: Uninstall packages.
uv pip tree: View the dependency tree for the environment.

## Locking packages in an environment (replacing pip-tools):
uv pip compile: Compile requirements into a lockfile.
uv pip sync: Sync an environment with a lockfile.

## Utility Managing and inspecting uv's state, such as the cache, storage directories, or performing a self-update:
uv cache clean: Remove cache entries.
uv cache prune: Remove outdated cache entries.
uv cache dir: Show the uv cache directory path.
uv tool dir: Show the uv tool directory path.
uv python dir: Show the uv installed Python versions path.
uv self update: Update uv to the latest version.

# **Ollama**
https://github.com/ollama/ollama

# **spacy**
https://spacy.io/usage

        source venv/bin/activate
            uv pip install -U pip setuptools wheel
            uv pip install -U 'spacy[cuda12x]'
            python -m spacy download pl_core_news_sm

# **Psycopg 3**
https://www.psycopg.org/

uv add psycopg[binary]

# **scikit-learn**
https://scikit-learn.org/stable/index.html#

uv add -U scikit-learn

# **Langchain**
https://python.langchain.com/docs/introduction/

uv add langchain
uv add -U langchain-ollama


# **SentenceTransformers**
https://sbert.net/docs/quickstart.html#sentence-transformer

uv add sentence-transformers

# **torch** 
https://pytorch.org/get-started/locally/
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129

# **cudnn** TODO ?
https://developer.nvidia.com/cudnn
https://github.com/NVIDIA/cudnn-frontend

# **TensorRT** TODO ?
https://docs.nvidia.com/tensorrt/index.html
https://github.com/NVIDIA/TensorRT

# DBeaver
## Main
    Host: localhost
    Port: 5432
    Database:
    Username:
    Password:
**Test connection**
## SSH
    Host:
    Port:22
    User Name:
    Authentication Method: Public key: .ssh\id_rsa
**Test tunnel configuration**



# GIT link
https://github.com/timescale/pgai/blob/main/docs/vectorizer/document-embeddings.md
https://github.com/TsLu1s/talknexus/blob/main/README.md

# Visual Studio Code
https://stackoverflow.com/questions/55310734/how-to-add-more-indentation-in-the-visual-studio-code-explorer-file-tree-structu

# PATH - OS
https://docs.python.org/3/library/pathlib.html#comparison-to-the-os-and-os-path-modules

# Networkx 
https://lopezyse.medium.com/knowledge-graphs-from-scratch-with-python-f3c2a05914cc

# Pystempel
https://htmlpreview.github.io/?https://github.com/dzieciou/pystempel/blob/master/Evaluation.html

# Streaming LLM
https://til.simonwillison.net/llms/streaming-llm-apis
https://www.youtube.com/watch?v=cy6EAp4iNN4

https://github.com/NeuralNine/youtube-tutorials/blob/main/LLM%20Development/main.py