from nltk.corpus import wordnet

# composition

print('car\'s synonym groups : ', wordnet.synsets('car'))
print()

car = wordnet.synset('car.n.01')

print('car.n.01 definition (for human) : ', car.definition())
print()

print('car.n.01 elements : ', car.lemma_names())
print()

print('car\'s hierarchy : ', car.hypernym_paths()[0])
print()


# similarity

car = wordnet.synset('car.n.01')
novel = wordnet.synset('novel.n.01')
dog = wordnet.synset('dog.n.01')
motocycle = wordnet.synset('motorcycle.n.01')

print('car & novel similarity : ', car.path_similarity(novel))
print('car & dog similarity : ', car.path_similarity(dog))
print('car & motocycle similarity : ', car.path_similarity(motocycle))