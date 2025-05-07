import os
from pathlib import Path
from tqdm import tqdm
from google.adk.agents import Agent
from extractor import extract_text_from_pdf, get_pdf_files
from chuncker import process_file
from embedder import build_faiss_index, load_chunks_from_dir
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict

model = SentenceTransformer("all-MiniLM-L6-v2")

def process_pdf_pipeline():
    """
    Process the PDF pipeline. Extract text from PDFs, chunk the text, and create embeddings.

    Args:
        None
    """
    # Get the directory of the current script
    current_dir = Path(__file__).parent

    # Define directories
    books_dir = current_dir / "books"
    shelf_dir = current_dir / "shelf"
    db_dir = current_dir / "db"
    embedded_dir = current_dir / "embedded"

    # Create necessary directories
    for dir_path in [shelf_dir, db_dir, embedded_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Step 1: Extract text from PDFs
    print("\n📚 Step 1: Extracting text from PDFs")
    pdf_files = get_pdf_files(str(books_dir))

    for pdf_path in tqdm(pdf_files, desc="Extracting PDFs"):
        text = extract_text_from_pdf(pdf_path)
        output_filename = Path(pdf_path).stem + ".txt"
        output_path = os.path.join(shelf_dir, output_filename)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

    # Step 2: Chunk the extracted text
    print("\n✂️ Step 2: Chunking extracted text")
    txt_files = list(shelf_dir.glob("*.txt"))

    for txt_file in tqdm(txt_files, desc="Chunking files"):
        process_file(str(txt_file), str(db_dir))

    # Step 3: Create embeddings and FAISS indices
    print("\n🔢 Step 3: Creating embeddings and indices")

    # Collect all chunks from all folders
    all_chunks = []
    for folder_name in tqdm(os.listdir(db_dir), desc="Loading chunks"):
        folder_path = os.path.join(db_dir, folder_name)
        if os.path.isdir(folder_path):
            chunks = load_chunks_from_dir(folder_path)
            all_chunks.extend(chunks)

    # Create single index and mapping for all chunks
    index_path = os.path.join(embedded_dir, "all_books.faiss")
    mapping_path = os.path.join(embedded_dir, "all_books.pkl")

    build_faiss_index(all_chunks, index_path, mapping_path)

    print("\n✅ Pipeline completed successfully!")


def load_faiss_index(index_path: str, mapping_path: str) -> Tuple[faiss.IndexFlatL2, List[str]]:
    """
    Load the FAISS index and mapping.

    Args:
        index_path (str): The path to the FAISS index.
        mapping_path (str): The path to the mapping.

    Returns:
        Tuple[faiss.IndexFlatL2, List[str]]: The FAISS index and mapping.
    """
    index = faiss.read_index(index_path)
    with open(mapping_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def search_chunks(query: str, index: faiss.IndexFlatL2, chunks: List[str], k: int = 5) -> List[str]:
    """
    Search the FAISS index for the top k chunks.

    Args:
        query (str): The query to search for.
        index (faiss.IndexFlatL2): The FAISS index.
        chunks (List[str]): The chunks to search from.
        k (int): The number of chunks to return.

    Returns:
        List[str]: The top k chunks.
    """
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]

def interactive_search():
    """
    Interactive search for the top k chunks.

    Args:
        None
    """

    current_dir = Path(__file__).parent
    embedded_dir = current_dir / "embedded"

    index_path = os.path.join(embedded_dir, "all_books.faiss")
    mapping_path = os.path.join(embedded_dir, "all_books.pkl")

    print("\n🔍 Loading search index...")
    index, chunks = load_faiss_index(index_path, mapping_path)
    print("✅ Search index loaded!")

    while True:
        user_query = input("\nAsk something about the books (type 'exit' to quit): ")
        if user_query.lower() in ("exit", "quit"):
            break

        results = search_chunks(user_query, index, chunks, k=5)
        print("\n🔍 Top matching chunks:\n")
        for i, res in enumerate(results):
            print(f"--- Chunk {i+1} ---\n{res[:500]}...\n")  # Limit preview length

# if __name__ == "__main__":
#     # Run the pipeline first
#     process_pdf_pipeline()

#     # Then start interactive search
#     interactive_search()


