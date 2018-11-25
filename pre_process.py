import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
#from stopword import pre_proces
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

train_text = "i love you "
sample_text = "i hate you and love you"

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
#tokenized = custom_sent_tokenizer.tokenize(sample_text)
stop_words = set(stopwords.words('english'))
word = word_tokenize(sample_text)
filtered_sentance = []

def pre_process():
    for i in word:
        if i not in stop_words:
            filtered_sentance.append(i)
    print(filtered_sentance)

    try:
        for i in filtered_sentance:
            words = nltk.word_tokenize(i)
            tagged =nltk.pos_tag(words)
            print(tagged)
    except exception as a:
        print(str(a))

pre_process()
#pre_proces()


