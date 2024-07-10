# Tonal Graph Harmonic Analyzer 

This algorithm offers an automatic analysis for Roman numerals.

## Installation
Get started by cloning the repository: 
```bash
git clone https://github.com/Sebastien-li/Tonal-Graph-Analyzer-Light
cd Tonal-Graph-Analyzer-Light
```

Get started by creating a virtual environment.
```bash
python -m venv venv
```
Windows:
```bash
venv/Scripts/activate.bat
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
### Analysis
To perform an analysis on all .mxl files in the data folder. To analyze your pieces, create a new .mxl file in 'data/{COMPOSER}/{TITLE}/score.mxl. You can also provide a romantext file to compare the results.

```bash
python -m main
```
