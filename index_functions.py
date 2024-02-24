# Standard Library Imports
from openai import OpenAI
import json
import nltk
import string
import requests
import os
import re
from uuid import uuid4
from typing import IO, Any, Dict, List, Tuple
from copy import deepcopy
import requests
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from pinecone.core.client.model.query_response import QueryResponse
from pinecone_text.sparse import BM25Encoder

# Third-Party Imports
import streamlit as st
import langchain
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

INDEX_NAME = "hybrid-search"

# Function to load in data found in the 'data' folder of the central repository; To upload your own data, simply remove the existing data in that folder and upload your own. Don't forget to update the prompt below!
@st.cache_data(show_spinner=True)
def load_data():
    with st.spinner(text="Loading and indexing the data. This shouldn't take more than a minute.."):
        # Download our files to the /content/ dir 
        github_dir = "https://github.com/pinecone-io/examples/raw/master/learn/generation/rag-for-hybrid/"
        filenames = ["freshdiskann_paper.pdf", "hnsw_paper.pdf", "ivfpq_paper.pdf"]
        for f in filenames:
            get_pdf(github_dir, f)
        return filenames    

#################################################################################
# Additional, specific functions I had in the Innovation CoPilot for inspiration:

def get_pdf(base_url: str, filename: str):
    """
    Download and write a PDF file from a github repository.

    :param url: URL of Github repository containing the file you want to download & write locally.
    """
    res = requests.get(base_url+filename)
    # Check if the request was successful (HTTP status code 200)
    if res.status_code == 200:
      with open("./content/"+filename, 'wb') as f:
          f.write(res.content)
          print(f"PDF downloaded and saved as {filename}")
    else:
      print(f"Failed to download the PDF. HTTP status code: {res.status_code}")

# Partition all of our PDFs and store their partitions in a dictionary for easy retrieval & inspection later
# Note: This takes a few mins to run (~12 mins; will be faster if running locally (~3 mins))

@st.cache_resource
def partition_files(filenames):
    # Read in our file paths
    freshdisk = os.path.join("./content/", filenames[0])
    hnsw = os.path.join("./content/", filenames[1])
    ivfpq = os.path.join("./content/", filenames[2])
    # Partition all of our PDFs and store their partitions in a dictionary for easy retrieval & inspection later
    # Note: This takes a few mins to run (~12 mins; will be faster if running locally (~3 mins))
    partitioned_files = {
        "freshdisk": partition_pdf(freshdisk, url=None, strategy = 'ocr_only'),
        "hnsw": partition_pdf(hnsw, url=None, strategy = 'ocr_only'),
        "ivfpq": partition_pdf(ivfpq, url=None, strategy = 'ocr_only'),
    }
    # Make an archived copy of partitioned_files dict so if we mess it up while cleaning, we don't have to re-ocr our PDFs:
    partitioned_files_copy = deepcopy(partitioned_files)
    # You can see in the preview above that each of our PDFs now has elements classifying different parts of the text, such as Text, Title, and EmailAddress.
    # Data cleaning matters a lot when it comes to hybrid search, because for the keyword-search part we care about each individual token (word).
    # Let's filter out all of the email addresses to start with, since we don't need those for any reason.
    # Remove unwanted EmailAddress category from dictionary of partitioned PDFs
    remove_unwanted_categories(partitioned_files, 'EmailAddress')
    print(partitioned_files.get('freshdisk'))
    print([i.text for i in partitioned_files.get('freshdisk')])
    remove_space_and_single_partitions(partitioned_files)
    rejoin_split_words(partitioned_files)
    remove_inline_citation_numbers(partitioned_files)
    stitch_partitions_back_together(partitioned_files)
    # Let's save our cleaned files to a new variable that makes more sense w/the current state
    cleaned_files = partitioned_files
    chunk_documents(cleaned_files)
    chunked_files = cleaned_files
    # Now that we have our chunks, we can create dense embeddings for each of them.
    # We'll use the 'all-MiniLM-L12-v2' model hosted by HuggingFace to create our dense embeddings. It's currently high on their MTEB (Massive Text Embedding Benchmark) Leaderboard (Reranking section), so it's a pretty safe bet. This will output dense vectors of 384 dimensions.
    # Make sure to save your chunks and embeddings (both sparse and dense) in pkl files, so that you don't have to wait for the embeddings to generate again if you want to rerun any steps in this notebook.
    freshdisk_dembeddings = produce_embeddings(chunked_files.get('freshdisk'))  # these take ~30s min to run
    hnsw_dembeddings = produce_embeddings(chunked_files.get('hnsw'))
    ivfpq_dembeddings = produce_embeddings(chunked_files.get('ivfpq'))
    
    # We can confirm the shape of each our dense embeddings is 384:

    # Make binary lists to keep track of any shapes that are *not* 384
    freshdisk_assertion = [0 for i in freshdisk_dembeddings if i.shape == 384]
    hnsw_assertion = [0 for i in hnsw_dembeddings if i.shape == 384]
    ivfpq_assertion = [0 for i in ivfpq_dembeddings if i.shape == 384]

    # Sum up our lists. If there are any embeddings that are not of shape 384, these sums will be > 0
    assert sum(freshdisk_assertion) == 0
    assert sum(hnsw_assertion) == 0
    assert sum(ivfpq_assertion) == 0

    # Create Sparse Embeddings of our Chunks
    #  We will use the BM25 algorithm to create our sparse embeddings. The resulting vector will represent an inverted index of the tokens in our chunks, constrained by things like chunk length.
    # Since we're using a ML-implemented version of BM25, we need to "fit" the model to our corpus. To do this, we'll combine all 3 of our PDFs together, so that the BM25 model can compute all the token frequencies etc correctly. We'll then encode each of our documents with our "fitted" model.
    # Join the content of all our PDFs together into 1 large corpus

    corpus = ""

    for i, v in chunked_files.items():
        corpus += ' '.join(v)
    print(len(corpus))  # Awesome, we've got lots o' tokens here for our BM25 model to learn :))
    # Initialize BM25 and fit to our corpus

    bm25 = st.session_state.session_state['bm25']
    bm25.fit(corpus)  

    # Create embeddings for each chunk
    freshdisk_sembeddings = [bm25.encode_documents(i) for i in chunked_files.get('freshdisk')]
    hnsw_sembeddings = [bm25.encode_documents(i) for i in chunked_files.get('hnsw')]
    ivfpq_sembeddings = [bm25.encode_documents(i) for i in chunked_files.get('ivfpq')]
    # We want the # of chunks per PDF to be equal to the # of sparse embeddings we've generated. Let's check that:
    assert len(freshdisk_sembeddings) == len(chunked_files.get('freshdisk'))
    assert len(hnsw_sembeddings) == len(chunked_files.get('hnsw'))
    assert len(ivfpq_sembeddings) == len(chunked_files.get('ivfpq'))

    # Getting Our Embeddings into Pinecone
    pinecone = st.session_state.session_state['pinecone']
    
    # create Pinecone index
    if INDEX_NAME not in [index.name for index in pinecone.list_indexes()]:
        # pinecone.delete_index(INDEX_NAME)
        pinecone.create_index(name=INDEX_NAME, 
                              dimension=384, # must match the dimensionality of our (dense) vectors
                              metric='dotproduct',
            spec=ServerlessSpec(cloud='aws', region='us-west-2'))
        print(pinecone.describe_index(INDEX_NAME))
        index = pinecone.Index(INDEX_NAME)
        freshdisk_ids = create_ids(chunked_files.get('freshdisk'))
        hnsw_ids = create_ids(chunked_files.get('hnsw'))
        ivfpq_ids = create_ids(chunked_files.get('ivfpq'))
        # Let's make sure we have the same # of IDs as there are chunks:
        assert len(freshdisk_ids) == len(chunked_files.get('freshdisk'))
        assert len(hnsw_ids) == len(chunked_files.get('hnsw'))
        assert len(ivfpq_ids) == len(chunked_files.get('ivfpq'))
        freshdisk_metadata = create_metadata_objs(chunked_files.get('freshdisk'))
        hnsw_metadata = create_metadata_objs(chunked_files.get('hnsw'))
        ivfpq_metadata = create_metadata_objs(chunked_files.get('ivfpq'))
        freshdisk_com_objs = create_composite_objs(freshdisk_ids, freshdisk_sembeddings, freshdisk_dembeddings, freshdisk_metadata)
        hnsw_com_objs = create_composite_objs(hnsw_ids, hnsw_sembeddings, hnsw_dembeddings, hnsw_metadata)
        ivfpq_com_objs = create_composite_objs(ivfpq_ids, ivfpq_sembeddings, ivfpq_dembeddings, ivfpq_metadata)
        index.upsert(freshdisk_com_objs)
        index.upsert(hnsw_com_objs)
        index.upsert(ivfpq_com_objs)
        print(index.describe_index_stats())
        
        return index





def remove_unwanted_categories(elements: Dict[str, List[Text]], unwanted_cat: str) -> None:
    """
    Remove partitions containing an unwanted category.

    :parameter elements: Partitioned pieces of our documents.
    :parameter unwanted_cat: The name of the category we'd like filtered out.
    """
    for key, value in elements.items():
        elements[key] = [i for i in value if not i.category == unwanted_cat]

# Remove empty spaces & single-letter/-digit partitions:

def remove_space_and_single_partitions(elements: Dict[str, List[Text]]) -> None:
    """
    Remove empty partitions & partitions with lengths of 1.

    :parameter elements: Partitioned pieces of our documents.
    """
    for key, value in elements.items():
        elements[key] = [i for i in value if len(i.text.strip()) > 1 ]

def remove_inline_citation_numbers(elements: Dict[str, List[Text]]) -> None:
    """
    Remove inline citation numbers from partitions.

    :parameter elements: Partitioned pieces of our documents.
    """
    for key, value in elements.items():
        pattern = re.compile(r'\[\s*(\d+\s*,\s*)*\d+\s*\]')
        elements[key] = [pattern.sub('', i) for i in value]

# Sew our partitions back together, per PDF:

def stitch_partitions_back_together(elements: Dict[str, List[Text]]) -> None:
    """
    Stitch partitions back into single string object.

    :parameter elements:  Partitioned pieces of our documents.
    """
    for key, value in elements.items():
        elements[key] = ' '.join(value)

def generate_chunks(doc: str, chunk_size: int = 512, chunk_overlap: int = 35) -> List[Document]:
    """
    Generate chunks of a certain size and token overlap.

    :param doc: Document we want to turn into chunks.
    :param chunk_size: Desired size of our chunks, in tokens (words).
    :param chunk_overlap: Desired # of tokens (words) that will overlap across chunks.

    :return: Chunks representations of the given document.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    return splitter.create_documents([doc])

def chunk_documents(docs: Dict[str, List[Text]],  chunk_size: int = 512, chunk_overlap: int = 35) -> None:
    """
    Iterate over documents and chunk each one.

    :parameter docs: The documents we want to chunk.
    :param chunk_size: Desired size of our chunks, in tokens (words).
    :param chunk_overlap: Desired # of tokens (words) that will overlap across chunks.
    """
    for key, value in docs.items():
        chunks = generate_chunks(value)
        docs[key] = [c.page_content for c in chunks]  # Grab the text representation of the chunks via the `page_content` attribute

def produce_embeddings(chunks: List[str]) -> List[str]:
    """
    Produce dense embeddings for each chunk.

    :param chunks: The chunks we want to create dense embeddings of.

    :return: Dense embeddings produced by our SentenceTransformer model `all-MiniLM-L12-v2`.
    """
    model = SentenceTransformer('all-MiniLM-L12-v2')
    embeddings = []
    for c in chunks:
        embedding = model.encode(c)
        embeddings.append(embedding)
    return embeddings

# Note: this function transforms our elemenets into their text representations
def rejoin_split_words(elements: Dict[str, List[Text]]) -> None:
    """
    Rejoing words that are split over pagebreaks.

    :parameter elements: Partitioned pieces of our documents.
    """
    for key, value in elements.items():
        elements[key] = [i.text.replace('- ', '') for i in value if '- ' in i.text]

def create_ids(chunks: str) -> List[str]:
    """
    Create unique IDs for each document (chunk) in our index.

    :param chunks: Chunks of our PDF file.

    :return: Unique IDs for chunks.
    """
    return [str(uuid4()) for _ in range(len(chunks))]

def create_metadata_objs(doc: List[str]) -> List[dict[str]]:
    """
    Create objects to store as metadata alongside our sparse and dense vectors in our hybird Pinecone index.

    :param doc: Chunks of a document we'd like to use while creating metadata objects.

    :return: Metadata objects with a "text" key and a value that points to the text content of each chunk.
    """
    return [{'text': d} for d in doc]

def create_composite_objs(ids: str, sembeddings: List[Dict[str, List[Any]]], dembeddings: List[float], metadata: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Create objects for indexing into Pinecone. Each object contains a document ID (which corresponds to the chunk, not the larger document),
    the chunk's sparse embedding, the chunk's dense embedding, and the chunk's corresponding metadata object.

    :param ids: Unique ID of a chunk we want to index.
    :param sembeddings: Sparse embedding representation of a chunk we want to index.
    :param dembeddings: Dense embedding representation of a chunk we want to index.
    :param metadata: Metadata objects with a "text" key and a value that points to the text content of each chunk.

    :return: Composite objects in the correct format for ingest into Pinecone.
    """
    to_index = []

    for i in range(len(metadata)):
        to_index_obj = {
                'id': ids[i],
                'sparse_values': sembeddings[i],
                'values': dembeddings[i],
                'metadata': metadata[i]
            }
        to_index.append(to_index_obj)
    return to_index

def weight_by_alpha(sparse_embedding: Dict[str, List[Any]], dense_embedding: List[float], alpha: float) -> Tuple[Dict[str, List[Any]], List[float]]:
    """
    Weight the values of our sparse and dense embeddings by the parameter alpha (0-1).

    :param sparse_embedding: Sparse embedding representation of one of our documents (or chunks).
    :param dense_embedding: Dense embedding representation of one of our documents (or chunks).
    :param alpha: Weighting parameter between 0-1 that controls the impact of sparse or dense embeddings on the retrieval and ranking
        of returned docs (chunks) in our index.

    :return: Weighted sparse and dense embeddings for one of our documents (chunks).
    """
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    hsparse = {
        'indices': sparse_embedding['indices'],
        'values':  [v * (1 - alpha) for v in sparse_embedding['values']]
    }
    hdense = [v * alpha for v in dense_embedding]
    return hsparse, hdense

def issue_hybrid_query(index, sparse_embedding: Dict[str, List[Any]], dense_embedding: List[float], alpha: float, top_k: int) -> QueryResponse:
    """
    Send properly formatted hybrid search query to Pinecone index and get back `k` ranked results (ranked by dot product similarity, as
        defined when we made our index).

    :param sparse_embedding: Sparse embedding representation of one of our documents (or chunks).
    :param dense_embedding: Dense embedding representation of one of our documents (or chunks).
    :param alpha: Weighting parameter between 0-1 that controls the impact of sparse or dense embeddings on the retrieval and ranking
        of returned docs (chunks) in our index.
    :param top_k: The number of documents (chunks) we want back from Pinecone.

    :return: QueryResponse object from Pinecone containing top-k results.
    """
    scaled_sparse, scaled_dense = weight_by_alpha(sparse_embedding, dense_embedding, alpha)

    result = index.query(
        vector=scaled_dense,
        sparse_vector=scaled_sparse,
        top_k=top_k,
        include_metadata=True
    )
    return result

