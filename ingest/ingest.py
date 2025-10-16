import os, glob
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

def load_env():
    load_dotenv()
    for k in ["OPENAI_API_KEY","PINECONE_API_KEY","PINECONE_INDEX_NAME","PINECONE_CLOUD","PINECONE_REGION","EMBEDDING_MODEL"]:
        if not os.getenv(k):
            raise RuntimeError(f"Missing environment variable: {k}")

def load_documents(docs_dir="/app/docs"):
    docs = []
    print(f"Current working directory: {os.getcwd()}")
    
    # Ensure we're using absolute paths
    abs_docs_dir = os.path.abspath(docs_dir)
    print(f"Looking for documents in: {abs_docs_dir}")
    
    if not os.path.exists(abs_docs_dir):
        print(f"Directory does not exist: {abs_docs_dir}")
        raise RuntimeError(f"Directory does not exist: {abs_docs_dir}")
    
    try:
        contents = os.listdir(abs_docs_dir)
        print(f"Directory contents of {abs_docs_dir}:", contents)
    except Exception as e:
        print(f"Error listing directory: {e}")
        raise
    
    # Use abs_docs_dir for all glob operations
    pdf_files = glob.glob(os.path.join(abs_docs_dir, "**/*.pdf"), recursive=True)
    txt_files = glob.glob(os.path.join(abs_docs_dir, "**/*.txt"), recursive=True)
    md_files = glob.glob(os.path.join(abs_docs_dir, "**/*.md"), recursive=True)
    
    print(f"Found PDF files: {pdf_files}")
    print(f"Found TXT files: {txt_files}")
    print(f"Found MD files: {md_files}")
    
    for p in pdf_files:
        docs.extend(PyPDFLoader(p).load())
    for p in txt_files:
        docs.extend(TextLoader(p, autodetect_encoding=True).load())
    for p in md_files:
        docs.extend(TextLoader(p, autodetect_encoding=True).load())
    
    if not docs:
        raise RuntimeError(f"No .pdf/.txt/.md documents found in {abs_docs_dir}/")
    return docs

def embedding_dim(model: str) -> int:
    if model == "text-embedding-3-small": return 1536
    if model == "text-embedding-3-large": return 3072
    raise ValueError(f"Unknown embedding dimension; please extend embedding_dim for model: {model}")

def ensure_index(index_name, dimension, cloud, region):
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    names = [i.name for i in pc.list_indexes()]
    
    # Delete existing index if dimensions don't match
    if index_name in names:
        existing_index = pc.describe_index(index_name)
        if existing_index.dimension != dimension:
            print(f"Deleting index with mismatched dimension: {index_name} (current_dim={existing_index.dimension}, required_dim={dimension})")
            pc.delete_index(index_name)
            names.remove(index_name)
    
    if index_name not in names:
        print(f"Creating Pinecone index: {index_name} (dim={dimension}, metric=cosine, {cloud}/{region})")
        pc.create_index(
            name=index_name, dimension=dimension, metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region)
        )
    else:
        print(f"Index already exists: {index_name}")

def main():
    load_env()
    index_name = os.environ["PINECONE_INDEX_NAME"]
    cloud      = os.environ["PINECONE_CLOUD"]
    region     = os.environ["PINECONE_REGION"]
    emb_model  = os.environ["EMBEDDING_MODEL"]

    # Use sentence-transformers model instead of OpenAI
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    dim = 384  # all-MiniLM-L6-v2 has 384 dimensions
    ensure_index(index_name, dim, cloud, region)

    # Use absolute path for docs directory
    docs = load_documents("/app/docs")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=120,
        separators=["\n\n","\n","。","！","？","，"," ", ""],
        add_start_index=True
    )
    chunks = splitter.split_documents(docs)
    for d in chunks:
        src = d.metadata.get("source") or d.metadata.get("file_path")
        d.metadata["source"] = str(src)
        d.metadata.setdefault("tag", "general")

    print(f"Number of chunks to upsert: {len(chunks)}")
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
    )
    print("✅ Vector ingestion complete!")

if __name__ == "__main__":
    main()