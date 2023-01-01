from nltk.corpus import wordnet as wn
#import nltk
#nltk.download('omw-1.4')
print(wn.synsets("would"))
print(wn.synsets("would")[0].pos())
dog = wn.synset(wn.synsets("would")[0].name())
print(dog.hyponyms())
print(dog.hypernyms())