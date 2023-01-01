## Virtual env
- Commands that were run to setup virtual env:
pip install -U pip setuptools wheel
pip install -U nltk
pip install -U numpy
nltk.download('stopwords')
nltk.download('wordnet')
pip install fuzzywuzzy
pip install python-Levenshtein


## INSTALL
python3 install.py
## RUN
- To run the program:
python3 qa_v3.py <inputfile.txt>

## TOOLS used
fuzzywuzzy - https://github.com/seatgeek/thefuzz#readme
NLTK - https://www.nltk.org/index.html
(Wordnet) from NLTK
