from flask import Flask, request, render_template
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import heapq
import re
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import get_data

app = Flask(__name__, template_folder="view")

documents = get_data.master_data

def load_model(filepath):
    return joblib.load(filepath)

route = "/"
@app.route(route, methods=['GET'])
def index():
    return render_template("index.html", data={})

route = "/search"
@app.route(route, methods=['GET'])
def search():
    keyword = request.args.get("keyword")
    loaded_vectorizer = load_model("tfidf_document.joblib")
    loaded_tfidf_matrix = load_model("tfidf_matrix.joblib")

    text = keyword
    text = BeautifulSoup(text, "html.parser").get_text()
    text = text.lower()
    text = re.sub(r'[\n\t\r]', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('indonesian'))
    tokens = [token for token in tokens if token.lower() not in stop_words]

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    preprocessed_document = " ".join(tokens)

    query = preprocessed_document
    query_vector = loaded_vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vector, loaded_tfidf_matrix)

    best_match_indices = heapq.nlargest(5, range(len(similarity_scores[0])), similarity_scores[0].__getitem__)
    best_match_scores = [similarity_scores[0][index] for index in best_match_indices]

    r = zip(best_match_indices, best_match_scores)
    hasil = []
    for index, score in r:
        print(f"Dokumen ke: {index} memiliki kemiripan: {score*100}%")
        if score > 0:
            hasil.append(documents[index])
    
    data = {
        'search_query': keyword,
        'search_results': hasil,
    }
    return render_template("hasil.html", **data)


if __name__ == '__main__':
    app.run(host = '127.0.0.1', port = 5000, debug = True)