import sys
import subprocess

# implement pip as a subprocess:
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'nltk'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'fuzzywuzzy'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'python-Levenshtein'])

import nltk
nltk.download('stopwords')
nltk.download('wordnet')