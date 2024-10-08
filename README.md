# Tonal Graph Harmonic Analyzer

This algorithm offers an automatic analysis tool for Roman numerals.

## Installation
This algorithm requires Python 3.10 or 3.11.

Get started by cloning the repository:
```bash
git clone https://github.com/Sebastien-li/Tonal-Graph-Analyzer-Light
cd Tonal-Graph-Analyzer-Light
```

Create a virtual environment.
```bash
python -m venv venv
```
Activate the virtual environment.

Windows:
```bash
"venv/Scripts/activate.bat"
```
Linux / MacOS:
```bash
source venv/bin/activate
```
Install the necessary requirements:
```bash
pip install -r requirements.txt
```
Replace the version of xmlToM21.py by copying the provided file to *venv/Lib/site-packages/music21/musicxml*

## Usage
### Perform an analysis on .mxl files in the data folder

To analyze your pieces, create a new .mxl file *data/{COMPOSER}/{TITLE}/score.mxl*.

You can also provide a romantext file *data/{COMPOSER}/{TITLE}/analysis.txt* to compare the results.

```bash
python -m main # Analyze all the files in the data folder
python -m main -p 'path/to/score.mxl' # Analyze only one specific mxl file.
```

### Perform a visualization of the analysis

For plotting performance reasons, the scores should not be stored in the same directory as the global analysis. To visualize the analysis on your pieces, create a new directory *assets/scores/{COMPOSER}/{TITLE}*. Then, add the files: *score.mxl*, *image.png*. Adding *analysis.txt* will also compare the results.

> [!WARNING]
> Please use relatively small pieces (<50 measures) or extract only the desired part.
```bash
python -m app
```
Then click on [this](http://127.0.0.1:8050/).
