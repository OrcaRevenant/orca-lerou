from sentence_transformers import SentenceTransformer, util
import tiktoken
from typing import List
import os
from pathlib import Path
from .hash_utils import check_file_hash

model = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

MAX_TOKENS = 512
SIMILARITY_THRESHOLD = 0.7

def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a text.

    Args:
        text (str): The text to count the tokens of.

    Returns:
        int: The number of tokens in the text.

    """

    return len(tokenizer.encode(text))

def load_paragraphs(path: str) -> List[str]:
    """
    Load the paragraphs from a file.

    Args:
        path (str): The path to the file.

    Returns:
        list[str]: A list of paragraphs from the file.

    """

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return [p.strip() for p in content.split("\n\n") if p.strip()]

def semantic_chunk(paragraphs: List[str]) -> List[str]:
    """
    Semantic chunk the paragraphs from a file.

    Args:
        paragraphs (List[str]): A list of paragraphs from the file.

    Returns:
        list[str]: A list of chunks from the file.

    """

    embeddings = model.encode(paragraphs, convert_to_tensor=True)
    chunks = []
    current_chunk = []
    current_tokens = 0

    for i, paragraph in enumerate(paragraphs):
        para_tokens = count_tokens(paragraph)

        if para_tokens > MAX_TOKENS:
            # Force flush the current chunk
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_tokens = 0
            chunks.append(paragraph)
            continue

        if not current_chunk:
            current_chunk.append(paragraph)
            current_tokens = para_tokens
            continue

        # Check semantic similarity to last paragraph in current chunk
        last_embedding = model.encode(current_chunk[-1], convert_to_tensor=True)
        current_embedding = embeddings[i]
        similarity = util.cos_sim(last_embedding, current_embedding).item()

        if similarity >= SIMILARITY_THRESHOLD and current_tokens + para_tokens <= MAX_TOKENS:
            current_chunk.append(paragraph)
            current_tokens += para_tokens
        else:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [paragraph]
            current_tokens = para_tokens

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks

def process_file(input_path: str, output_dir: str) -> None:
    """
    Process a single file and save its chunks to a subfolder in the output directory.

    Args:
        input_path (str): The path to the file to process.
        output_dir (str): The path to the output directory.
    """
    # Get the base filename without extension
    base_name = Path(input_path).stem

    # Create a subfolder for this file's chunks
    file_output_dir = os.path.join(output_dir, base_name)
    os.makedirs(file_output_dir, exist_ok=True)

    # Check if the file has been processed before
    if os.path.exists(file_output_dir) and check_file_hash(input_path, "chunker"):
        print(f"⏭️ Skipping {input_path} - no changes detected")
        return

    # Load and chunk the file
    paragraphs = load_paragraphs(input_path)
    chunks = semantic_chunk(paragraphs)

    # Save each chunk to a separate file in the subfolder
    for i, chunk in enumerate(chunks):
        output_path = os.path.join(file_output_dir, f"chunk_{i+1}.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(chunk)

def process_shelf() -> None:
    """
    Process all .txt files in the shelf directory and save chunks to db directory.

    Args:
        None
    """
    # Get the directory of the current script
    current_dir = Path(__file__).parent

    # Define input and output directories
    shelf_dir = current_dir / "shelf"
    db_dir = current_dir / "db"

    # Process each .txt file in the shelf directory
    for txt_file in shelf_dir.glob("*.txt"):
        print(f"Processing {txt_file.name}...")
        process_file(str(txt_file), str(db_dir))
        print(f"Finished processing {txt_file.name}")

if __name__ == "__main__":
    process_shelf()
