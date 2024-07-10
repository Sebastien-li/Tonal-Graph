# Tonal Graph Harmonic Analyzer 

This algorithm offers an automatic analysis for Roman numerals.

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

## Usage
### Perform an analysis on all .mxl files in the data folder

To analyze your pieces, create a new .mxl file *data/{COMPOSER}/{TITLE}/score.mxl*. 

You can also provide a romantext file *data/{COMPOSER}/{TITLE}/analysis.txt* to compare the results.

```bash
python -m main
```
