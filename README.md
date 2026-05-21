# Spectrum Matching Project

## Getting Started

This project uses `uv` for dependency management. Follow the instructions below to set up and run the project.

### Prerequisites

- Python 3.13 or higher
- `uv` package manager ([install uv](https://docs.astral.sh/uv/getting-started/installation/))

### Installation

1. Clone or navigate to the project directory:
   ```bash
   cd spectrum-matching-project
   ```

2. Install dependencies using `uv`:
   ```bash
   uv sync
   ```

   This command will:
   - Read the `pyproject.toml` file
   - Install all dependencies specified in the project
   - Create a virtual environment (if needed)
   - Lock dependencies in `uv.lock`

### Running the Project

To run the main script:
```bash
uv run python main.py
```

Or directly execute Python code in the virtual environment:
```bash
uv run python <your_script.py>
```

### Useful `uv` Commands

- **List installed packages**: `uv pip list`
- **Add a new dependency**: `uv add <package_name>`
- **Remove a dependency**: `uv remove <package_name>`
- **Update dependencies**: `uv sync --upgrade`
- **Run with specific Python version**: `uv run --python 3.13 python main.py`

For more information about `uv`, visit the [official documentation](https://docs.astral.sh/uv/).

# TODO: how to add data to project