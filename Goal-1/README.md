GOAL 1  - Load word vectors from FastText and explore similar words. Use Annoy or plain NumPy. Plot similar words. Generate your own word vectors on a specific dataset.
DUE: 26/June/2024
Deliverable: Notebook in your Github repository that showcases your work related to this task.

Reading
papers
https://arxiv.org/pdf/1301.3781v3 
https://nlp.stanford.edu/pubs/glove.pdf 
blogs
https://making.lyst.com/2014/11/11/word-embeddings-for-fashion/
https://multithreaded.stitchfix.com/blog/2015/03/11/word-is-worth-a-thousand-vectors/
https://stackoverflow.blog/2023/11/09/an-intuitive-introduction-to-text-embeddings/

Data
A collection of pre-trained word vectors can be found here https://fasttext.cc/docs/en/english-vectors.html and https://nlp.stanford.edu/projects/glove/

Code 
Working with Text https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
Nearest Neighbors https://scikit-learn.org/stable/modules/neighbors.html#neighbors
Word2Vec https://radimrehurek.com/gensim/
Loading word vectors https://github.com/blester125/word-vectors
Annoy https://github.com/spotify/annoy
This Kaggle Notebook https://www.kaggle.com/code/pierremegret/gensim-word2vec-tutorial shows how you can use Gensim to train your own word embeddings.

End to end Kaggle notebook to load fasttext vectors, index them and visualize them 
https://www.kaggle.com/code/harshsinghal/rit-bootcamp-june-2024-goal-1

Raw Data
AWS Case Studies Dataset https://www.kaggle.com/datasets/harshsinghal/aws-case-studies-and-blogs

Google Cloud Case Studies Dataset (case study files from 1600+ customers on how they use various Google Cloud Services)
https://www.kaggle.com/datasets/harshsinghal/google-cloud-customer-case-studies/data

Ideas
Can you extract keywords and named entities from the AWS case studies? What interesting visuals can you show on similar words from the AWS case studies dataset. 
Which companies are using what AWS technologies and how?

Using pre-trained word embeddings can you create a sentiment classification model? 
https://www.kaggle.com/datasets/ahmedabdulhamid/reviews-dataset
See https://arxiv.org/pdf/1607.01759

Approach
For each review in the Positive reviews dataset, break up the review into words.
For each word, extract the embedding vector from fasttext
At the review level, average the embeddings for all the words contained in the review.
Now you have a single vector for a review in the Positives reviews dataset.
Repeat steps 1 to 4 for the negative reviews dataset.
Now you have vectors and a label (positive or negative).
Apply KNN classification.
Train a simple classifier - logistic regression, random forest, xgboost use scikit-learn for these algorithms
For the reviews in the test dataset, apply the per word embedding lookup and average and score the review using the trained classifier in step 7.
Plot ROC curve to identify a suitable cut-off that will give you the best performance (tradeoff between precision and recall). 
Train a simple fully connected neural network using Keras for the above classification task and evaluate performance.
