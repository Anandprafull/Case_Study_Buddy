# Project README

## Goal 1: Word Vectors Exploration and Application

### Task Description
- Load word vectors from FastText and explore similar words using Annoy or NumPy.
- Plot similar words and generate your own word vectors on a specific dataset.

### Due Date
- 26th June 2024

### Deliverable
- Notebook in your GitHub repository that showcases your work related to this task.

## Resources

### Reading
- **Papers:**
  - [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781v3)
  - [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)
  
- **Blogs:**
  - [Word Embeddings for Fashion](https://making.lyst.com/2014/11/11/word-embeddings-for-fashion/)
  - [A Word is Worth a Thousand Vectors](https://multithreaded.stitchfix.com/blog/2015/03/11/word-is-worth-a-thousand-vectors/)
  - [An Intuitive Introduction to Text Embeddings](https://stackoverflow.blog/2023/11/09/an-intuitive-introduction-to-text-embeddings/)

### Data
- A collection of pre-trained word vectors:
  - [FastText](https://fasttext.cc/docs/en/english-vectors.html)
  - [GloVe](https://nlp.stanford.edu/projects/glove/)
- **Code:**
  - [Working with Text Data in scikit-learn](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
  - [Nearest Neighbors in scikit-learn](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)
  - [Word2Vec in Gensim](https://radimrehurek.com/gensim/)
  - [Loading Word Vectors](https://github.com/blester125/word-vectors)
  - [Annoy Library](https://github.com/spotify/annoy)
- **Kaggle Notebook:** [Gensim Word2Vec Tutorial](https://www.kaggle.com/code/pierremegret/gensim-word2vec-tutorial)

### End-to-End Example
- [Kaggle Notebook to Load FastText Vectors](https://www.kaggle.com/code/harshsinghal/rit-bootcamp-june-2024-goal-1)

### Raw Data
- **AWS Case Studies Dataset:** [AWS Case Studies and Blogs](https://www.kaggle.com/datasets/harshsinghal/aws-case-studies-and-blogs)
- **Google Cloud Case Studies Dataset:** [Google Cloud Customer Case Studies](https://www.kaggle.com/datasets/harshsinghal/google-cloud-customer-case-studies/data)

## Ideas for Exploration
- Extract keywords and named entities from the AWS case studies.
- Visualize similar words from the AWS case studies dataset.
- Analyze which companies are using specific AWS technologies and how they are using them.

## Additional Tasks
- Create a sentiment classification model using pre-trained word embeddings:
  - Dataset: [Reviews Dataset](https://www.kaggle.com/datasets/ahmedabdulhamid/reviews-dataset)
  - Reference: [A Supervised Approach to Sentiment Analysis](https://arxiv.org/pdf/1607.01759)

### Approach
1. For each review in the Positive reviews dataset, break it into words.
2. Extract the embedding vector from FastText for each word.
3. Average the embeddings to get a single vector for each positive review.
4. Repeat steps 1-3 for the negative reviews dataset.
5. Apply KNN classification.
6. Train simple classifiers (e.g., logistic regression, random forest, xgboost) using scikit-learn.
7. For reviews in the test dataset, apply per-word embedding lookup and average.
8. Score the review using the trained classifier.
9. Plot ROC curve to find the best performance cutoff (precision vs. recall tradeoff).
10. Train a simple fully connected neural network using Keras for the classification task and evaluate performance.


 ## Next steps 

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

---

This README provides a structured overview of your project's goals, resources, datasets, and the approach you plan to take. Make sure to update it as you progress and achieve milestones in your project.
