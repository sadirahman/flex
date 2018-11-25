from nltk.corpus import wordnet


syns = wordnet.synsets("program")

#synset
#print(syns[0])


#print(syns[0].lemmas())

#Just the word
print(syns[0].lemmas()[0].name())

#definition
#print(syns[0].definition())

#example
print(syns[0].examples())

synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))

w1 = wordnet.synset("traffic.n.01")
w2 = wordnet.synset("aggregation.n.01")
print(w1.wup_similarity(w2))

w1 = wordnet.synset("conveyance.n.01")
w2 = wordnet.synset("vehicle.n.01")
print(w1.wup_similarity(w2))

w1 = wordnet.synset("road.n.01")
w2 = wordnet.synset("transportation.n.01")
print(w1.wup_similarity(w2))

w1 = wordnet.synset("cancer.n.01")
w2 = wordnet.synset("blood.n.01")
print(w1.wup_similarity(w2))

w1 = wordnet.synset("condition.n.01")
w2 = wordnet.synset("disease.n.01")
print(w1.wup_similarity(w2))

w1 = wordnet.synset("cell.n.01")
w2 = wordnet.synset("compartment.n.01")
print(w1.wup_similarity(w2))

w1 = wordnet.synset("spread.n.01")
w2 = wordnet.synset("expanse.n.01")
print(w1.wup_similarity(w2))

w1 = wordnet.synset("time.n.01")
w2 = wordnet.synset("event.n.01")
print(w1.wup_similarity(w2))

w1 = wordnet.synset("drug.n.01")
w2 = wordnet.synset("medicine.n.01")
print(w1.wup_similarity(w2))

w1 = wordnet.synset("expert.n.01")
w2 = wordnet.synset("ability.n.01")
print(w1.wup_similarity(w2))

w1 = wordnet.synset("sister.n.01")
w2 = wordnet.synset("unit.n.01")
print(w1.wup_similarity(w2))


