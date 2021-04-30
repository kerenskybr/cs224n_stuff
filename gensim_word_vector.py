import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# conda install -c conda-forge gensim
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

#glove_file = datapath('/home/roger/Documents/stanford_CS224N/corpora/glove.6B.100d.txt')
#word2vec_glove_file = (get_tmpfile("glove.6B.100d.txt"))
glove_file = datapath('/home/roger/Documents/stanford_CS224N/corpora/glove_s100_portugues.txt')
word2vec_glove_file = (get_tmpfile("glove_s100_portugues.txt"))
glove2word2vec(glove_file, word2vec_glove_file)

model = KeyedVectors.load_word2vec_format(word2vec_glove_file)

# print(model.most_similar('trip'))
# print(model.most_similar(negative='banana'))
# result = model.most_similar(positive=['woman', 'king'], negative=['man'])
# print("{}: {:.4f}".format(*result[0]))

def analogy(x1, x2, y1):
    result = model.most_similar(positive= [y1, x2], negative=[x1])
    return result[0][0]

print(analogy('brasil', 'brasileiro', 'japão'))

def display_pca_scatterplot(model, words=None, sample=0):
    if words == None:
        if sample > 0:
            words = np.random.choice(list(model.vocab.keys()), sample)
        else:
            words = [ word for word in model.vocab ]
        
    word_vectors = np.array([model[w] for w in words])

    twodim = PCA().fit_transform(word_vectors)[:,:2]
    
    plt.figure(figsize=(6,6))
    plt.scatter(twodim[:,0], twodim[:,1], edgecolors='k', c='r')
    for word, (x,y) in zip(words, twodim):
        plt.text(x+0.05, y+0.05, word)

    plt.show()

display_pca_scatterplot(model, 
                        ['café', 'chá', 'cerveja', 'vinho', 'brandy', 'rum', 'champanhe', 'água',
                         'espagueti', 'hamburger', 'pizza', 'falafel', 'sushi', 'meatballs',
                         'cachorro', 'cavalo', 'gato', 'macaco', 'papagaio', 'koala', 'lagarto',
                         'sapo', 'rã', 'macaco', 'canguru', 'morcego', 'lobo',
                         'frança', 'alemanha', 'hungria', 'australia', 'fiji', 'china',
                         'tema', 'tarefa', 'problema', 'exame', 'teste', 'aula',
                         'escola', 'colégio', 'universidade', 'instituto'])

display_pca_scatterplot(model, sample=300)

'''
print(analogy('japan', 'japanese', 'brazil'))
# Japan > Japanese | Brazil > ???

print(model.doesnt_match("breakfast cereal dinner lunch".split()))

print(analogy('man', 'king', 'woman'))
# Man > King | Woman > ???

print(analogy('australia', 'beer', 'france'))

def display_pca_scatterplot(model, words=None, sample=0):
    if words == None:
        if sample > 0:
            words = np.random.choice(list(model.vocab.keys()), sample)
        else:
            words = [ word for word in model.vocab ]
        
    word_vectors = np.array([model[w] for w in words])

    twodim = PCA().fit_transform(word_vectors)[:,:2]
    
    plt.figure(figsize=(6,6))
    plt.scatter(twodim[:,0], twodim[:,1], edgecolors='k', c='r')
    for word, (x,y) in zip(words, twodim):
        plt.text(x+0.05, y+0.05, word)

    plt.show()

display_pca_scatterplot(model, 
                        ['coffee', 'tea', 'beer', 'wine', 'brandy', 'rum', 'champagne', 'water',
                         'spaghetti', 'borscht', 'hamburger', 'pizza', 'falafel', 'sushi', 'meatballs',
                         'dog', 'horse', 'cat', 'monkey', 'parrot', 'koala', 'lizard',
                         'frog', 'toad', 'monkey', 'ape', 'kangaroo', 'wombat', 'wolf',
                         'france', 'germany', 'hungary', 'luxembourg', 'australia', 'fiji', 'china',
                         'homework', 'assignment', 'problem', 'exam', 'test', 'class',
                         'school', 'college', 'university', 'institute'])

display_pca_scatterplot(model, sample=300)
'''