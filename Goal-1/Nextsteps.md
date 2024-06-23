 ## Next steps after goal 1 solution

# Step 1: Install necessary libraries
!pip install gensim annoy matplotlib

# Step 2: Download and load FastText word vectors
!wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
!unzip wiki-news-300d-1M.vec.zip

from gensim.models import KeyedVectors
from annoy import AnnoyIndex
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Step 3: Load word vectors into memory
model_path = 'wiki-news-300d-1M.vec'
model = KeyedVectors.load_word2vec_format(model_path)

# Step 4: Build Annoy index
vector_size = 300
index = AnnoyIndex(vector_size, 'angular')

for i, word in enumerate(model.index_to_key):
    index.add_item(i, model[word])

index.build(10)

# Step 5: Define functions to get and visualize similar words
def get_similar_words(word, top_n=10):
    word_id = model.key_to_index[word]
    similar_word_ids = index.get_nns_by_item(word_id, top_n)
    return [model.index_to_key[id] for id in similar_word_ids]

def visualize_similar_words(word):
    similar_words = get_similar_words(word)
    word_vectors = [model[word]] + [model[w] for w in similar_words]
    
    tsne = TSNE(n_components=2, random_state=42)
    word_vectors_2d = tsne.fit_transform(word_vectors)
    
    plt.figure(figsize=(10, 10))
    plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1], edgecolors='k', c='r')
    
    for i, word in enumerate([word] + similar_words):
        plt.text(word_vectors_2d[i, 0], word_vectors_2d[i, 1], word)
    
    plt.title(f"Similar Words to '{word}'")
    plt.xlabel("TSNE Component 1")
    plt.ylabel("TSNE Component 2")
    plt.show()

# Step 6: Example usage
visualize_similar_words('king')
