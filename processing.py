from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
lemmatizer = WordNetLemmatizer()
def dataPreprocessing(mssg):
    words = word_tokenize(mssg)
    words = [lemmatizer.lemmatize(w.lower())  for w in words if w.lower() not in stopwords.words('english')]
    mssg = ' '.join(words)
    return vectorizer.transform([mssg]).toarray()
