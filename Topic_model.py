from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
import nltk
from nltk.corpus import wordnet
import gensim
import pyLDAvis.gensim as gensimvis
import pyLDAvis
import warnings

warnings.filterwarnings('ignore')

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
l_lemma = WordNetLemmatizer()

# create sample documents
# doc_a = "a motor vehicle with four wheels; usually propelled by an internal combustion engine"
# doc_b = "a wheeled vehicle adapted to the rails of railroad"
# doc_c = "the compartment that is suspended from an airship and that carries personnel and the cargo and the power plant"
# doc_d = "where passengers ride up and down"
# doc_e = "a conveyance for passengers or freight on a cable railway"

doc_a = "a motor vehicle with four wheels; usually propelled by an internal combustion engine"
doc_b = "a class of problems. Algorithms can perform calculation, data processing and automated reasoning tasks."
doc_c = "A vehicle is a machine that transports people or cargo. Vehicles include wagons, bicycles, motor vehicles, railed vehicles, watercraft, amphibious vehicles, aircraft and spacecraft."
doc_d = "doctor is a professional who practises medicine, which is concerned with promoting, maintaining, or restoring health through the study, diagnosis, and treatment of disease, injury, and other physical and mental impairments."
doc_e = "A university is an institution of higher education and research which awards academic degrees in various academic disciplines. Universities typically provide undergraduate education and postgraduate education."

doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]

# list for tokenized documents in loop
texts = []

# loop through document list
for i in doc_set:
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]

    # stem tokens
    stemmed_tokens = [l_lemma.lemmatize(i) for i in stopped_tokens]

    #     pos_tagger = [nltk.pos_tag(i) for i in stemmed_tokens]

    #     nn_tagged = [(word,tag) for word, tag in pos_tagger
    #                 if tag.startswith('NN')]

    # add tokens to list

    texts.append(stemmed_tokens)

l = []
m = []

# for i in texts:
a = nltk.pos_tag(texts[0])
nn_tagged = [(word, tag) for word, tag in a if tag.startswith('NN') or tag.startswith('NNP')]

for i in nn_tagged:
    m.append(i[0])
l.append(m)

n = []
a = nltk.pos_tag(texts[1])
nn_tagged = [(word, tag) for word, tag in a if tag.startswith('NN') or tag.startswith('NNP')]

for i in nn_tagged:
    n.append(i[0])
l.append(n)

o = []
a = nltk.pos_tag(texts[2])
nn_tagged = [(word, tag) for word, tag in a if tag.startswith('NN') or tag.startswith('NNP')]

for i in nn_tagged:
    o.append(i[0])
l.append(o)

p = []
a = nltk.pos_tag(texts[3])
nn_tagged = [(word, tag) for word, tag in a if tag.startswith('NN') or tag.startswith('NNP')]

for i in nn_tagged:
    p.append(i[0])
l.append(p)

q = []
a = nltk.pos_tag(texts[4])
nn_tagged = [(word, tag) for word, tag in a if tag.startswith('NN') or tag.startswith('NNP')]

for i in nn_tagged:
    q.append(i[0])
l.append(q)
print(l)

# print(m)

# l=[]
#

# a=nltk.pos_tag(texts[0])
# nn_tagged = [(word,tag) for word, tag in a
#                 if tag.startswith('NN') or tag.startswith('NNP')]

# for i in nn_tagged:
#     l.append(i[0])

# print(l)
# ss=[l]
# print(ss)

# m=[]
# a=nltk.pos_tag(texts[1])
# nn_tagged = [(word,tag) for word, tag in a
#                 if tag.startswith('NN') or tag.startswith('NNP')]

# for i in nn_tagged:
#     m.append(i[0])

# print(m)
# print(ss+=m)


# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(l)

# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in l]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=4, id2word=dictionary, passes=2000, random_state=1)
# ldamodel2 =  gensim.models.LdaMulticore(corpus,
#                                    num_topics = 4,
#                                    id2word = dictionary,
#                                    passes = 2000,
#                                    workers = 2)

# print(nltk.pos_tag(texts[0]))
# a=nltk.pos_tag(texts[0])
# nn_tagged = [(word,tag) for word, tag in a
#                 if tag.startswith('NN') or tag.startswith('NNP')]
# print(nn_tagged)
# l=[]
# for i in nn_tagged:
#     l.append(i[0])
# print(l)

# l=[]
# m=[]
# for i in texts:
#     a=nltk.pos_tag(i)
#     nn_tagged = [(word,tag) for word, tag in a
#                 if tag.startswith('NN') or tag.startswith('NNP')]

# #     abc=[a for i in nn_tagged]
# #     l.append(abc)

#     for i in nn_tagged:
#         l.append(i[0])


# print(l)


# for i in texts:
#     nn_vb_tagged = [(word,tag) for word, tag in i
#                 if tag.startswith('NN') or tag.startswith('NNP')]
#     print(nn_vb_tagged)
# print(pos_tagger)
# print(tokens)
# print(texts)
# print(dictionary)
# print(ldamodel)
# print(corpus)
v = ldamodel.print_topics(num_topics=4, num_words=2)
print("LDA Model Clustering result: \n",v)