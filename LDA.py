doc1 = "sweet dangerous to eat. My brother likes to have sweet, but not my mother."
doc2 = "My mother consumes a lot of time pushing my brother about to dance exercise."
doc3 = "Doctors recommend that driving may produce improved stress and blood pressure."
doc4 = "my father never seems to drive my brother to do better."
doc5 = "Health experts say that sweet is bad for your health."

# compile documents
doc_complete = [doc1, doc2, doc3, doc4, doc5]
print(doc_complete)
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import PunktSentenceTokenizer
import string
import warnings
warnings.filterwarnings('ignore')
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())

    return normalized

doc_clean = [clean(doc).split() for doc in doc_complete]
print(doc_clean)

# Importing Gensim
import gensim
from gensim import corpora


# Creating the term dictionary of our courpus, where every unique term is assigned an index.
dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50,random_state=1)

result = ldamodel.print_topics(num_topics=3, num_words=5)
print(result)
#print(result[0][1].split())

import re
from nltk.corpus import wordnet



first = result[0][1].split()[0]
f_w = re.split("[^a-zA-Z]*",first)
second = result[1][1].split()[0]
s_w = re.split("[^a-zA-Z]*",second)

print(f_w)
print(s_w)

syns = wordnet.synsets(best_level)
des=""

for i in f_w:
    best_level=str(i)
    print(best_level)
    des.app(syns[0].definition())



for i1 in s_w:
    print(str(i1))


