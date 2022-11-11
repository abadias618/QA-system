## Virtual env
- Commands that were run to setup virtual env:
pip install -U pip setuptools wheel
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install transformers==4.7
pip install datasets

## RUN
- To run the program:

python3 qa.py <inputfile.txt>