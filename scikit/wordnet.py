import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet')

print(wn.synsets('dog'))
dog = wn.synset('dog.n.01')

print(dog.hypernyms())
print(dog.hyponyms())
