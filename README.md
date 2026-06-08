# Medical Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers medical questions using a local LLaMA model and Pinecone vector database.

---

## How it works

1. A medical PDF is loaded, split into chunks, and embedded using `all-MiniLM-L6-v2`
2. The chunks are stored in a Pinecone vector index
3. When a user asks a question, the top 4 relevant chunks are retrieved
4. A local LLaMA 3.2 model generates an answer grounded in the retrieved context
5. The answer is served via a Flask web interface

---

## Project structure

```
Medical-Chatbot/
│
├── data/
│   └── medical_book.pdf          # Source medical knowledge base
│
├── models/
│   └── Llama-3.2-3B-Instruct.gguf  # Local quantised LLaMA model (Q4)
│
├── src/
│   ├── helper.py                 # load_pdf, split_text, get_embeddings
│   ├── rag_chain.py              # Full RAG chain (retriever + LLM + prompt)
│   └── prompt.py                 # SYSTEM_PROMPT template
│
├── templates/
│   └── chat.html                 # Jinja2 HTML template (Flask frontend)
│
├── static/
│   └── style.css                 # Page styling
│
├── store_index.py                # One-time script: embed PDF → upload to Pinecone
├── app.py                        # Flask web server
├── requirements.txt              # Python dependencies
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/medical-chatbot.git
cd medical-chatbot
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the root folder:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=medical-chatbot
```

### 5. Add the LLaMA model

Download `Llama-3.2-3B-Instruct-Q4_K_M.gguf` and place it in the `models/` folder.

> You can download it from [Hugging Face](https://huggingface.co/models) — search for `Llama-3.2-3B-Instruct-GGUF`.

### 6. Index the medical PDF (run once)

```bash
python store_index.py
```

This reads `data/medical_book.pdf`, splits it into chunks, embeds each chunk using `all-MiniLM-L6-v2`, and uploads the vectors to Pinecone.

### 7. Start the chatbot

```bash
python app.py
```

Open your browser at `http://127.0.0.1:5000`

---

## Key components

### `src/helper.py`

```python
def load_pdf(path):
    # reads PDF → returns List[Document] (one per page)

def split_text(documents):
    # splits into 1000-char chunks with 200-char overlap

def get_embeddings():
    # loads all-MiniLM-L6-v2 → produces 384-dim vectors
```

### `src/rag_chain.py`

```python
chain = (
    RunnableParallel({
        "context": retriever | format_docs,  # fetch top 4 chunks
        "input": RunnablePassthrough()        # pass question through
    })
    | prompt        # fill context + question into template
    | llm           # local LLaMA generates the answer
    | StrOutputParser()
)
```

### `app.py`

```python
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        question = request.form["question"]
        answer = chain.invoke(question)   # RAG pipeline runs here
    return render_template("chat.html", question=question, answer=answer)
```

---

## Tech stack

| Component | Technology |
|-----------|------------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | Pinecone (Serverless, AWS us-east-1) |
| LLM | LLaMA 3.2 3B Instruct (Q4 quantised, llama.cpp) |
| Framework | LangChain + Flask |
| Frontend | Jinja2 + plain CSS |

---

## Requirements

```
flask
python-dotenv
langchain
langchain-community
langchain-huggingface
langchain-pinecone
langchain-core
pinecone-client
sentence-transformers
llama-cpp-python
pypdf
```

---

## Notes

- `store_index.py` only needs to be run **once**. Re-running it will upload duplicate vectors.
- The LLaMA model runs **entirely locally** — no OpenAI API key needed.
- `temperature=0.3` keeps answers factual; increase it for more creative responses.
- `k=4` retrieves the 4 most relevant chunks — increase for broader context.