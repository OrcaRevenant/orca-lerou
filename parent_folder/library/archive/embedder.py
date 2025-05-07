import faiss
import os
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from hash_utils import check_file_hash

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_chunks_from_dir(directory: str) -> list[str]:
    """
    Load the chunks from a directory.

    Args:
        directory (str): The directory to load the chunks from.

    Returns:
        list[str]: A list of chunks from the directory.
    """
    chunks = []
    for root, _, files in os.walk(directory):
        for filename in sorted(files):
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    chunks.append(f.read())
    return chunks

def build_faiss_index(chunks: list[str], index_path: str, mapping_path: str):
    """
    Build a FAISS index from a list of chunks.

    Args:
        chunks (list[str]): A list of chunks to build the index from.
        index_path (str): The path to save the index to.
        mapping_path (str): The path to save the mapping to.

    Returns:
        None
    """
    embeddings = model.encode(chunks, show_progress_bar=True)
    dimension = embeddings[0].shape[0]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Ensure embedded directory exists
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    faiss.write_index(index, index_path)

    with open(mapping_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"✅ FAISS index saved to {index_path}")
    print(f"✅ Chunk mapping saved to {mapping_path}")

def process_all_books():
    """
    Process all books in the db directory and create embeddings.

    Args:
        None
    """
    db_dir = os.path.join(os.path.dirname(__file__), "db")
    embedded_dir = os.path.join(os.path.dirname(__file__), "embedded")

    # Create embedded directory if it doesn't exist
    os.makedirs(embedded_dir, exist_ok=True)

    # Process each book folder separately
    for folder_name in tqdm(os.listdir(db_dir), desc="Creating embeddings"):
        folder_path = os.path.join(db_dir, folder_name)
        if os.path.isdir(folder_path):
            print(f"\nProcessing {folder_name}...")

            # Check if the folder has been processed before
            index_path = os.path.join(embedded_dir, f"{folder_name}.faiss")
            if os.path.exists(index_path) and check_file_hash(folder_path, "embedder"):
                print(f"⏭️ Skipping {folder_name} - no changes detected")
                continue

            chunks = load_chunks_from_dir(folder_path)
            if chunks:
                mapping_path = os.path.join(embedded_dir, f"{folder_name}.pkl")
                build_faiss_index(chunks, index_path, mapping_path)
            else:
                print(f"No text files found in {folder_name}")

if __name__ == "__main__":
    process_all_books()
