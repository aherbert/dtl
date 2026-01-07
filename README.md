# DtL

Distance-to-lamina analysis.

## Installation

```bash
# Install uv
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Clone the repository
git clone https://github.com/aherbert/dtl.git
# Change into the project directory
cd dtl
# Create and activate virtual environment
uv sync
source .venv/bin/activate
```

## Updating

```bash
# Pull the latest changes
git pull
# Resync the virtual environment
uv sync
```

## Usage

### Cellpose models

Segmentation uses `cellpose` which requires that the named model be installed in the
`cellpose` models directory. For custom models this can be achieved using:

        cellpose --add_model [model path]

The default cellpose 3 model is `cyto3` (no  install required). This works well for
typical nuclei images.

### Analysis of images

Analysis of images requires a CYX input image. This can be in TIFF or ICS
(Image Cytometry Standard) format. The script `dtl-analysis.py` is used to run the analysis:

```bash
# Activate the environment (if not active)
source .venv/bin/activate

# Analyse
./dtl-analysis.py /path/to/image.[tiff|ics]
```

## Development

This project uses [pre-commit](https://pre-commit.com/) to create actions to validate
changes to the source code for each `git commit`.
Install the hooks for your development repository clone using:

    pre-commit install

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
