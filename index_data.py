from google.cloud import storage
from google.cloud import documentai_v1 as documentai
from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import InternalServerError, RetryError
import re

from typing import List, Optional
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

from google.cloud import aiplatform

import os
import csv

# Set up your Google Cloud project and credentials
project_id = "affable-seat-433116-u6"
bucket_name = "ride-data-dev"
vector_bucket_name = "ride-vector-data"
processor_id = "c9c1464cf5a00c84"  # Replace with your Document AI processor ID
location = "us"

gcs_input_prefix = f"gs://{bucket_name}/"
gcs_output_uri = f"gs://{bucket_name}/processed/"
gcs_vector_uri = f"gs://{vector_bucket_name}/batch_root/"

batch_root = 'batch_root/'

task: str = "RETRIEVAL_DOCUMENT"
model_name: str = "text-embedding-004"
dimensionality: Optional[int] = 384

chunk_size = 900
overlap = 150

# Initialize storage and Document AI clients
storage_client = storage.Client()
documentai_client = documentai.DocumentProcessorServiceClient()

def batch_process_documents():
    # Set the API endpoint for the specified location
    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")

    # Specify a GCS URI Prefix to process an entire directory
    gcs_prefix = documentai.GcsPrefix(gcs_uri_prefix=gcs_input_prefix)
    input_config = documentai.BatchDocumentsInputConfig(gcs_prefix=gcs_prefix)

    # Cloud Storage URI for the Output Directory
    gcs_output_config = documentai.DocumentOutputConfig.GcsOutputConfig(
        gcs_uri=gcs_output_uri
    )

    # Where to write results
    output_config = documentai.DocumentOutputConfig(gcs_output_config=gcs_output_config)

    # The full resource name of the processor, e.g.:
    # projects/{project_id}/locations/{location}/processors/{processor_id}
    name = documentai_client.processor_path(project_id, location, processor_id)

    request = documentai.BatchProcessRequest(
        name=name,
        input_documents=input_config,
        document_output_config=output_config,
    )

    # BatchProcess returns a Long Running Operation (LRO)
    operation = documentai_client.batch_process_documents(request)

    # Continually polls the operation until it is complete.
    # This could take some time for larger files
    try:
        print(f"Waiting for operation {operation.operation.name} to complete...")
        operation.result(timeout=120)
    # Catch exception when operation doesn't finish before timeout
    except (RetryError, InternalServerError) as e:
        print(e.message)

    # After the operation is complete,
    # get output document information from operation metadata
    metadata = documentai.BatchProcessMetadata(operation.metadata)

    if metadata.state != documentai.BatchProcessMetadata.State.SUCCEEDED:
        raise ValueError(f"Batch Process Failed: {metadata.state_message}")

    print("Output files:")
    # One process per Input Document
    documents = []
    for process in list(metadata.individual_process_statuses):
        print(process)
        # output_gcs_destination format: gs://BUCKET/PREFIX/OPERATION_NUMBER/INPUT_FILE_NUMBER/
        matches = re.match(r"gs://(.*?)/(.*)", process.output_gcs_destination)
        if not matches:
            print(
                "Could not parse output GCS destination:",
                process.output_gcs_destination,
            )
            continue

        output_bucket, output_prefix = matches.groups()

        # Get List of Document Objects from the Output Bucket
        output_blobs = storage_client.list_blobs(output_bucket, prefix=output_prefix)        

        # Document AI may output multiple JSON files per source file
        for blob in output_blobs:
            # Document AI should only output JSON files to GCS
            if blob.content_type != "application/json":
                print(
                    f"Skipping non-supported file: {blob.name} - Mimetype: {blob.content_type}"
                )
                continue

            # Download JSON File as bytes object and convert to Document Object
            print(f"Fetching {blob.name}")
            document = documentai.Document.from_json(
                blob.download_as_bytes(), ignore_unknown_fields=True
            )
            documents.append(document)
            # assembled = assemble_document(document)

    return documents
#documents[0].document_layout.blocks[0].
def assemble_document(doc):
    """
    Recursively assembles text from a documentai.Document, DocumentLayout, or DocumentLayoutBlock.    
    Args:
        doc: The document component to process. It can be an instance of:
             - documentai.Document
             - documentai.Document.DocumentLayout
             - documentai.Document.DocumentLayout.DocumentLayoutBlock    
    Returns:
        A string containing the assembled text from the document.
    """
    if isinstance(doc, documentai.Document):
        # Process the document's layout
        return assemble_document(doc.document_layout)    
    elif isinstance(doc, documentai.Document.DocumentLayout):
        # Accumulate text from all blocks within the layout
        text_parts = []
        for block in doc.blocks:
            text_parts.append(assemble_document(block))
        return ''.join(text_parts)    
    elif isinstance(doc, documentai.Document.DocumentLayout.DocumentLayoutBlock):
        text_parts = []
        text_block = doc.text_block        
        if text_block.text:
            text_parts.append(text_block.text)
            if "heading" in text_block.type_.lower():
                # Add two newlines after headings for separation
                text_parts = ["\n\n"] + text_parts
            else:
                # Add a space after regular text blocks
                text_parts.append(" ")        
        if text_block.blocks:
            # Recursively process nested blocks
            for block in text_block.blocks:
                text_parts.append(assemble_document(block))        
        return ''.join(text_parts)    
    else:
        # Unsupported document component
        return ""


def vectorize_documents(documents : list[str]):
    """Embeds texts with a pre-trained, foundational model."""
    split_documents = []
    for document in documents:
        # chunks = document.split("\n\n")
        chunks = [document[i:i+chunk_size] for i in range(0,len(document),overlap)]
        split_documents += chunks
    
    for i, document in enumerate(split_documents):
        save_snippet(str(i), document)

    model = TextEmbeddingModel.from_pretrained(model_name)
    inputs = [TextEmbeddingInput(text, task) for text in split_documents]
    kwargs = dict(output_dimensionality=dimensionality) if dimensionality else {}
    embeddings = model.get_embeddings(inputs, **kwargs)
    return [embedding.values for embedding in embeddings]

def vectorize_single_document(document : str, index : int):
    """Embeds texts with a pre-trained, foundational model."""
    split_document = [document[i:i+chunk_size] for i in range(0,len(document),overlap)]
        
    for i, snippet in enumerate(split_document):
        save_snippet(str(index + i), snippet)

    model = TextEmbeddingModel.from_pretrained(model_name)
    inputs = [TextEmbeddingInput(text, task) for text in split_document]
    kwargs = dict(output_dimensionality=dimensionality) if dimensionality else {}
    embeddings = model.get_embeddings(inputs, **kwargs)
    return [embedding.values for embedding in embeddings], index + len(split_document)

def store_embeddings(embeddings):
    bucket = storage_client.bucket(vector_bucket_name)
    # Prepare the data for CSV format
    data = []
    for i, embedding in enumerate(embeddings):
        record = {
            "id": str(i),
            "embedding": embedding  # Assuming embedding[0] contains the dense vector
        }
        data.append(record)  
    # Define file paths
    feature_file_path = os.path.join(batch_root, 'feature_file_1.csv')
    feature_file_blob = bucket.blob(feature_file_path)
    # Write the data to a CSV file
    with open('feature_file_1.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for record in data:
            writer.writerow([record['id']] + record['embedding'])
    # Upload the CSV to GCS
    feature_file_blob.upload_from_filename('feature_file_1.csv')
    os.remove('feature_file_1.csv')
    # Create or get an existing index
    index = aiplatform.MatchingEngineIndex(index_name=
                                           'projects/872489082435/locations/us-central1/indexes/1767887154114985984')
    # Wait for the index to be created
    index.wait()
    # Update embeddings
    index.update_embeddings(contents_delta_uri=gcs_vector_uri)

def save_snippet(snippet_id: str, snippet_text: str):
    """Saves a single snippet with its ID as the filename in GCS."""
    bucket = storage_client.bucket(vector_bucket_name)
    snippet_blob = bucket.blob(f'snippets/{snippet_id}.txt')
    snippet_blob.upload_from_string(snippet_text)

def index():
    try:
        documents = batch_process_documents()
        assembled = [assemble_document(doc) for doc in documents]
        vectors = vectorize_documents(assembled)
        store_embeddings(vectors)
        return "Success"
    except Exception as e:
        print(e)
        return "Error occurred"