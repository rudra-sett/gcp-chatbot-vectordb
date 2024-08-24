import os
from flask import Flask, request, jsonify

# Import the indexing and retrieval functions
from index_data import index
from retrieve_data import find_matches

app = Flask(__name__)

@app.route("/index", methods=["POST"])
def sync_and_index():
    """
    Endpoint to process documents, generate embeddings, and store them in the vector index.
    """
    result = index()
    if result == "Success":
        return jsonify({"message": "Indexing completed successfully"}), 200
    else:
        return jsonify({"message": "An error occurred during indexing"}), 500

@app.route("/retrieve", methods=["POST"])
def retrieve():
    """
    Endpoint to perform a vector search based on the provided query.
    """
    data = request.get_json()
    query = data.get("query", "")
    
    if not query:
        return jsonify({"message": "Query parameter is required"}), 400

    snippets = find_matches(query)

    if snippets:
        return jsonify({"snippets": snippets}), 200
    else:
        return jsonify({"message": "No matches found"}), 404

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
