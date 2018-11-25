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

# doc_a = "Cancer is actually a group of many related diseases that all have to do with cells. Cells are the very small units that make up all living things, including the human body."
# doc_b = "Cancer happens when cells that are not normal grow and spread very fast. Normal body cells grow and divide and know to stop growing. Over time, they also die."
# doc_c = "Cancer cells usually group or clump together to form tumors (say: TOO-mers). A growing tumor becomes a lump of cancer cells that can destroy the normal cells."
# doc_d = "This is how cancer spreads. The spread of a tumor to a new place in the body is called metastasis."
# doc_e = "It can take a while for a doctor to figure out a kid has cancer."

# doc_a = "Floods occur when  excessive water due to continued torrential rains. Normally dry lands may also get flooded when water overflows the banks of water, or when there is heavy and continuous rainfall. "
# doc_b = "Floods also occur in modern cities and towns where there is a high density of human population, and increased urban development by way of housing and other construction."
# doc_c = "In such areas trees are also felled for construction activity. Poor drainage systems in modern cities also lead to flooding when there are continuous or heavy rains."
# doc_d = "Felling of trees leads to increased soil erosion, and silting of water bodies that also leads to flooding during heavy rains."
# doc_e = "Floods cause loss to life and property. Humans , animals and birds are likely to perish in a flood."

# doc_a = "Drug addiction is a dependence syndrome. It is a condition where a person feels a strong desire to consume drugs and can’t do without them."
# doc_b= "The feeling to consume it is more important for them than other daily chores and even their family"
# doc_c= "If the addicted one does not use it for longer time he is likely to feel depressed and isolated."
# doc_d= "Addiction is the state where mind and body just cannot do without it."
# doc_e= "The brain changes are persistent that is why drug addiction is often defined as a form of mental disorder."

doc_a = "Sugar is bad to consume. My sister likes to have sugar, but not my father."
doc_b = "My father spends a lot of time driving my sister around to dance practice."
doc_c = "Doctors suggest that driving may cause increased stress and blood pressure."
doc_d = "Sometimes I feel pressure to perform well at school, but my father never seems to drive my sister to do better."
doc_e = "Health experts say that Sugar is not good for your lifestyle."

# doc_a = "Traffic jam means a long line of vehicles that can not move or that can move very slowly. It is a common affair in the big cities of our country. There are many causes of traffic jam. Rapid growth of population and the increasing amount of vehicles are the main causes of it."
# doc_b = "Vehicles are much more than the roads can accommodate. The indiscriminate playing of rickshaw is another causes of it. Haphazard parking of vehicles alongside the pavement also causes of it. "
# doc_c = " Violation of traffic rules is also responsible for it. The drivers do not follow traffic rules. Traffic jam causes untold sufferings to people. Sometimes it raises our mental tension. It causes loss of our valuable time. We have to wait to reach our destination. "
# doc_d = " The students, the office-going people, the businessmen and the patients in the ambulance are the worst sufferers of it. Traffic jam can be removed by enforcing traffic jam strictly. "
# doc_e = "The narrow roads should be broadened. By pass roads should be constructed in the big towns. One way movement of vehicles and building of fly over can solve this problem. We can reduce it by raising public awareness."

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
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=50, random_state=1)
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
v = ldamodel.print_topics(num_topics=3, num_words=2)
print("1st lda ",v)
# t1,t2,t3,t4
# for i in v:
#     if i==0:
#         t1=i
#     elif i==1:
#         t2=i
#     elif i==2:
#         t3=i
#     elif i==3:
#         t4=i


t1 = v[0][1]
t2 = v[1][1]
t3 = v[2][1]
#t4 = v[3][1]
print(t1, "\n", t2, "\n", t3, "\n")

# print(t1)
tv2 = t1.split()[0]
# print(tv2)
import re

tv3 = " ".join(re.findall("[a-zA-Z]+", tv2))
# print(tv3)
tv5 = re.split("[^a-zA-Z]*", tv3)
# print(tv4)
best_topic = ""
for item in tv5:
    best_topic = str(item)
    #print(best_topic)

syns = wordnet.synsets(best_topic)
print(syns[0])
des = []

n_doc_a = ""
n_doc_b = ""
n_doc_c = ""
n_doc_d = ""
n_doc_e = ""

for i in range(1):
    if i == 0:
        n_doc_a = syns[i].definition()
        print(n_doc_a)
    elif i == 1:
        n_doc_b = syns[i].definition()
    elif i == 2:
        n_doc_c = syns[i].definition()
    elif i == 3:
        n_doc_d = syns[i].definition()
    elif i == 4:
        n_doc_e = syns[i].definition()

#     des.append(syns[i].definition())
#     des.append(".")
# print(des)

n_doc_set = [n_doc_a, n_doc_b, n_doc_c, n_doc_d, n_doc_e]
n_texts = []
for i in n_doc_set:
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]

    # stem tokens
    stemmed_tokens = [l_lemma.lemmatize(i) for i in stopped_tokens]

    n_texts.append(stemmed_tokens)

print(n_texts)

n_l = []
n_m = []

# for i in texts:
n_a = nltk.pos_tag(n_texts[0])
n_nn_tagged = [(word, tag) for word, tag in n_a if tag.startswith('NN') or tag.startswith('NNP')]

for i in n_nn_tagged:
    n_m.append(i[0])
n_l.append(n_m)

n_n = []
n_a = nltk.pos_tag(n_texts[1])
n_nn_tagged = [(word, tag) for word, tag in n_a if tag.startswith('NN') or tag.startswith('NNP')]

for i in n_nn_tagged:
    n_n.append(i[0])
n_l.append(n_n)

n_o = []
n_a = nltk.pos_tag(n_texts[2])
n_nn_tagged = [(word, tag) for word, tag in n_a if tag.startswith('NN') or tag.startswith('NNP')]

for i in n_nn_tagged:
    n_o.append(i[0])
n_l.append(n_o)

n_p = []
n_a = nltk.pos_tag(n_texts[3])
n_nn_tagged = [(word, tag) for word, tag in n_a if tag.startswith('NN') or tag.startswith('NNP')]

for i in n_nn_tagged:
    n_p.append(i[0])
n_l.append(n_p)

n_q = []
n_a = nltk.pos_tag(n_texts[4])
n_nn_tagged = [(word, tag) for word, tag in n_a if tag.startswith('NN') or tag.startswith('NNP')]

for i in n_nn_tagged:
    n_q.append(i[0])
n_l.append(n_q)
print(n_l)

n_dictionary = corpora.Dictionary(n_l)

# convert tokenized documents into a document-term matrix
n_corpus = [n_dictionary.doc2bow(n_text) for n_text in n_l]

# generate LDA model
n_ldamodel = gensim.models.ldamodel.LdaModel(n_corpus, num_topics=1, id2word=n_dictionary, passes=2000, random_state=1)
n_v = n_ldamodel.print_topics(num_topics=1, num_words=1)
print(n_v)
nn = n_v[0][1].split()[0]
n_v1 = re.split("[^a-zA-Z]*", nn)
print(n_v1)
print("Topic Label:")
for i in n_v1:
    level=str(i)
    print(level)
# noun =[]
# noun = n_l
# print(noun)
# w1 = wordnet.synset(best_topic)
# w2 = wordnet.synset("blood.n.01")
# print(w1.wup_similarity(w2))

# print(t2)
# print(t3)
# print(t4)
# print(ldamodel2.print_topics(num_topics=4, num_words=2))
#vis_data = gensimvis.prepare(ldamodel, corpus, dictionary)
#pyLDAvis.display(vis_data)