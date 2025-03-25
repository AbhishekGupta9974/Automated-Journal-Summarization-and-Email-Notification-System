from flask import Flask, request, jsonify
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Load the summarization model
summarizer = pipeline("summarization", model="D:/Spring Initializr/textSummarizationApi/final_model")

# Load sentence transformer model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

app = Flask(__name__)

@app.route('/summarize', methods=['POST'])
def summarize_text():
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Generate the summary
        summary = summarizer(text, max_length=150, min_length=30, do_sample=False)[0]["summary_text"]

        # Convert text and summary into embeddings
        text_embedding = embedding_model.encode(text, convert_to_tensor=True)
        summary_embedding = embedding_model.encode(summary, convert_to_tensor=True)

        # Compute cosine similarity
        similarity_score = util.pytorch_cos_sim(text_embedding, summary_embedding).item()
        accuracy = round(similarity_score * 100, 2)  # Convert to percentage

        return jsonify({
            "summary": summary,
            "accuracy": f"{accuracy}%"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

