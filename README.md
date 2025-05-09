# Book Search Agent

A Python-based book search agent that utilizes Google's ADK (Agent Development Kit) and Gemini 2.0 Flash model to search through books and return relevant chunks of information.

## Project Structure

```
.
├── parent_folder/
│   └── library/
│       ├── agent.py          # Main agent implementation
│       ├── __init__.py       # Package initialization
│       └── archive/          # Additional agent utilities
├── .venv/                    # Python virtual environment
└── LICENSE                   # Project license
```

## Dependencies

The project uses the following packages:

- google.adk (Agent Development Kit)
- gemini-2.0-flash model
- faiss (Facebook AI Similarity Search) - for efficient similarity search and clustering
- sentence-transformers - for text embeddings and semantic search
- tqdm - for progress bars
- pathlib - for path manipulation
- typing - for type hints
- pickle - for serialization
- hashlib - for file hashing
- json - for JSON data handling

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

2. Install the required dependencies:

```bash
pip install google-adk faiss-cpu sentence-transformers tqdm
```

## Usage

The project implements a book search agent that can:

- Search through books
- Return top k chunks of relevant information
- Process and analyze book content

The main agent is implemented in `parent_folder/library/agent.py` and uses the Gemini 2.0 Flash model for processing.

## License

This project is licensed under the terms included in the LICENSE file.

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request
