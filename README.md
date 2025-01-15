### PDF-Answering-RAG
This project demonstrates the process of creating a chatbot that leverages internal data sources, specifically PDF documents, to generate responses. The chatbot utilizes a Retrieval-Augmented Generation (RAG) architecture, incorporating vector embeddings from documents to provide informative and contextually relevant answers.

### Usage

To start this run this jupyter notebook.
### Architecture

1. **Data Loading**: Utilizing `PyPDFLoader` for reading PDFs.
2. **Text Splitting**: Chunking Text using `RecursiveCharacterTextSplitter` from Langchain.
3. **Vector Embedding**: Generating vector embeddings using `HuggingFace Embeddings`.
4. **LLM Implementation**: Finally using `FAISS DB` with Open Source `Llama 2-7b-chat-hf`.

### Tools and Libraries Used

- Jupyter Notebook: For interactive development and demonstrations.
- Python: The primary programming language.
- `langchain`, `langchain_community`, `sentence-tranformers` and `faiss-cpu`: For document processing, text splitting, embedding generation and storing indexes.

### Implementation Details

The notebook includes detailed code cells for each step of the process, from loading and processing documents to initializing and querying this RAG.
