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

Install the maven and Java prerequisites for the
[BioIO](https://bioio-devs.github.io/bioio/OVERVIEW.html) library to use BioFormats.
See [bioio-bioformats](https://github.com/bioio-devs/bioio-bioformats?tab=readme-ov-file#special-installation-instructions).
This requires a full install of a recent version of Java to allow the scyjava bridge from
Python to Java to have the correct libraries.

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

Analysis of images requires a CZYX input image. This can be in any format supported
by the [BioIO](https://bioio-devs.github.io/bioio/OVERVIEW.html) library, e.g. TIFF,
ICS (Image Cytometry Standard), or CZI (Carl Zeiss Image).
The script `dtl-analysis.py` is used to run the analysis:

```bash
# Activate the environment (if not active)
source .venv/bin/activate

# Analyse
./dtl-analysis.py /path/to/image.[tiff|ics|czi]
```

This script will perform the following steps:

1. Segment the selected nuclei channel.
1. Identify spots in the selected spot channel.
1. Identify spots in the selected lamina channel.
1. Find the closest distance between spots and the egde of the nucleus, or a lamina
region.

Results are saved to files with the same prefix as the input image:

- `.objects.tiff`: Nuclei label mask.
- `.spots.tiff`: Spot channel label mask.
- `.lamina.tiff`: Lamina channel label mask.
- `.spots.csv`: Spot details table.
- `.summary.csv`: Nuclei summary table.
- `.settings.json`: JSON file with the runtime settings.

The results can be visualised in `Napari` using the `--view` option. This will load
the image as channels and the 3 label layers. The results tables are associated with
the appropriate label layers. The editing tools within `Napari` can be used to update
the label masks, e.g. add or remove spots; update the nuclei objects. A widget
within `Napari` allows the results tables to be regenerated from modified labels. This
will save the current label layers to file allowing the analysis to be continued in
a subsequent session by reloading the results:

```bash
# [Re]Analyse and view
./dtl-analysis.py /path/to/image.[tiff|ics|czi] --view
```

Analysis can be repeated which will reload existing results or restart the analysis at the
given stage, e.g. 1; 2; or 3.

### Multiple images

Multiple images can be passed as arguments to the analysis script. It is also possible to
pass in directories. In this case the script will run on any file with the CZI or ICS extensions
and any TIFF file containing a CZYX image. This allows the script to run on a directory
containing existing result masks as these YX images will be ignored.

```bash
# Analyse an image and two image directories
./dtl-analysis.py /path/to/image.ics /path/to/images/ /path/to/more/images/ --view
```

### Reporting

The results CSV files can be collated and used to generate reports across the analysis.
This can be done by passing individual CSV files or results directories. Only files
ending `.spots.csv` or `.summary.csv` are loaded. Reports are printed to the console
and saved to the specified output directory. Reports can be selected or by default
all reports are generated. Use the help option (`-h`) to view parameters that change the
report queries.

```bash
# Generate reports on the DtL analysis
./dtl-reports.py /path/to/images/ /path/to/more/images/
```

## Development

This project uses [pre-commit](https://pre-commit.com/) to create actions to validate
changes to the source code for each `git commit`.
Install the hooks for your development repository clone using:

    pre-commit install

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
