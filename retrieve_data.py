from google.cloud import storage
from google.cloud import aiplatform_v1
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

# Set variables for the current deployed index.
API_ENDPOINT="723219736.us-central1-872489082435.vdb.vertexai.goog"
INDEX_ENDPOINT="projects/872489082435/locations/us-central1/indexEndpoints/5661671244441845760"
DEPLOYED_INDEX_ID="ride_index_1724456188226"

# Set variables for embeddings
task: str = "RETRIEVAL_DOCUMENT"
model_name: str = "text-embedding-004"
dimensionality = 384

# Set number of matches
k = 3

# Configure Vector Search client
client_options = {
  "api_endpoint": API_ENDPOINT
}
vector_search_client = aiplatform_v1.MatchServiceClient(
  client_options=client_options,
)

storage_client = storage.Client()

def find_matches(query : str):
  model = TextEmbeddingModel.from_pretrained(model_name)
  inputs = [TextEmbeddingInput(query, task)]
  kwargs = dict(output_dimensionality=dimensionality) if dimensionality else {}
  embeddings = model.get_embeddings(inputs, **kwargs)
  # Build FindNeighborsRequest object
  datapoint = aiplatform_v1.IndexDatapoint(
    feature_vector=embeddings[0].values
  )
  query = aiplatform_v1.FindNeighborsRequest.Query(
    datapoint=datapoint,
    # The number of nearest neighbors to be retrieved
    neighbor_count=10
  )
  request = aiplatform_v1.FindNeighborsRequest(
    index_endpoint=INDEX_ENDPOINT,
    deployed_index_id=DEPLOYED_INDEX_ID,
    # Request can have multiple queries
    queries=[query],
    return_full_datapoint=False,
  )
  # Execute the request
  response = vector_search_client.find_neighbors(request)
  # Handle the response
  print(response)
  top_k = response.nearest_neighbors[0].neighbors[:k]

  snippets = []

  for match in top_k:
      id = match.datapoint.datapoint_id
      for i in range(int(id)-1,int(id)+2):
        snippet = retrieve_snippet(i)
        snippets.append(snippet)
  
  return snippets

vector_bucket_name = "ride-vector-data"

def retrieve_snippet(snippet_id: str) -> str:
    """Retrieves a snippet from GCS by its ID."""
    bucket = storage_client.bucket(vector_bucket_name)
    snippet_blob = bucket.blob(f'snippets/{snippet_id}.txt')

    if snippet_blob.exists():
        return snippet_blob.download_as_text()
    else:
        return ""