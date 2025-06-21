from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import math

app = Flask(__name__)
CORS(app)

def preprocess_text(text):
    # Lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def tf(text):
    words = text.split()
    tf_dict = {}
    for w in words:
        tf_dict[w] = tf_dict.get(w, 0) + 1
    total_words = len(words)
    for w in tf_dict:
        tf_dict[w] /= total_words
    return tf_dict

def idf(texts):
    import math
    N = len(texts)
    idf_dict = {}
    all_words = set()
    for t in texts:
        all_words.update(t.split())
    for word in all_words:
        containing = sum(1 for t in texts if word in t.split())
        idf_dict[word] = math.log((N + 1) / (containing + 1)) + 1
    return idf_dict

def tfidf(text, idf_dict):
    tf_dict = tf(text)
    tfidf_dict = {}
    for w, val in tf_dict.items():
        tfidf_dict[w] = val * idf_dict.get(w, 0)
    return tfidf_dict

def cosine_similarity(vec1, vec2):
    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    all_words = set(vec1.keys()) | set(vec2.keys())
    for word in all_words:
        v1 = vec1.get(word, 0.0)
        v2 = vec2.get(word, 0.0)
        dot += v1 * v2
        norm1 += v1 * v1
        norm2 += v2 * v2
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (math.sqrt(norm1) * math.sqrt(norm2))

@app.route('/check', methods=['POST'])
def check_plagiarism():
    try:
        data = request.json
        text1 = data.get('text1', '')
        text2 = data.get('text2', '')

        # Preprocess
        text1 = preprocess_text(text1)
        text2 = preprocess_text(text2)

        # Calculate idf using both texts
        idf_dict = idf([text1, text2])
        tfidf1 = tfidf(text1, idf_dict)
        tfidf2 = tfidf(text2, idf_dict)

        similarity = cosine_similarity(tfidf1, tfidf2) * 100
        return jsonify({'similarity': round(similarity, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
