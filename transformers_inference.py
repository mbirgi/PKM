from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)
model = pipeline('feature-extraction', model='sentence-transformers/all-MiniLM-L6-v2')

@app.route('/inference', methods=['POST'])
def inference():
    data = request.json
    text = data['text']
    embeddings = model(text)
    return jsonify(embeddings)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)