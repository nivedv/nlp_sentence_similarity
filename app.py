from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#from sentence_transformers import Setnence

# sample sentece 
sentence1 = "this is an example sentece"
sentence2 = "another sentence for comparison"
# load pre-trained model 
#
#preprocess sentence
stop_words = set(stopwords.words('english'))
porter= PorterStemmer

def preprocess(sentece):
    words = word_tokenize(sentece.lower())
    words = [porter.stem(word) for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(words)
# create a bag-of-words vectors
preprocess_sentence1 = preprocess(sentence1)
preprocess_sentence2 = preprocess(sentence2)
vectorizer = CountVectorizer()
X= vectorizer.fit_transform([preprocess_sentence1,preprocess_sentence2])
# Calculate cosine similarity
cosine_sim = cosine_similarity(X[0],X[1])
print(f"Cosine similarity : {cosine_sim[0][0]}")
