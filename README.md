# Conversational RAG with Context-Aware Retrieval & Session History

A Retrieval-Augmented Generation (RAG) pipeline built using **LangChain Expression Language (LCEL)**, **Cohere Models**, and **Chroma Vector Database**. 

This repository demonstrates how to build a stateful conversational agent capable of accurately answering user queries over custom PDF document collections while dynamically retaining conversation history across multiple interaction turns.

---

## 🔑 Key Features

* **PDF Ingestion & Processing**: Uses `pdfminer.six` and LangChain's `RecursiveCharacterTextSplitter` for optimal chunk size management and metadata tagging.
* **Vector Embeddings & Storage**: High-dimensional text embeddings generated via `CohereEmbeddings` stored locally in a `Chroma` vector database.
* **History-Aware Retrieval**: Reformulates user queries dynamically based on previous context, converting follow-up questions into standalone queries before vector retrieval.
* **Session Management**: Built with `RunnableWithMessageHistory` to seamlessly maintain individual state and memory per `session_id`.

---

## 🛠️ Tech Stack & Dependencies

* **Framework**: LangChain (`langchain`, `langchain-community`, `langchain-classic`)
* **LLM & Embeddings**: Cohere (`langchain-cohere`, `embed-english-v3.0`, `command-r`)
* **Vector Database**: ChromaDB
* **Document Parser**: PDFMiner (`pdfminer.six`)

---

## 🚀 Getting Started

### 1. Prerequisites

Make sure you have a valid Cohere API Key. You can get a free trial key from [dashboard.cohere.com](https://dashboard.cohere.com).

### 2. Installation

Clone the repository and install the dependencies:

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
pip install -r requirements.txt
```

*Or install directly via pip:*

```bash
pip install langchain-cohere langchain pdfminer.six chromadb langchain-community langchain-text-splitters
```

### 3. API Key Setup

Set your Cohere API key as an environment variable:

```python
import os
os.environ["COHERE_API_KEY"] = "your-cohere-api-key"
```

---

## 💻 Usage & Code Workflow

### 1. Vector Database Initialization

```python
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import Chroma

persist_directory = "./chroma_db"
embedding = CohereEmbeddings(model="embed-english-v3.0", user_agent="langchain")

# Initialize VectorDB
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
```

### 2. Setting Up the History-Aware Chain

```python
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# LLM Setup
llm = ChatCohere(model="command-a-plus-05-2026", temperature=0)

# History-aware Retriever Setup
prompt_history_aware = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question, formulate a standalone question."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(
    llm, vectordb.as_retriever(), prompt_history_aware
)

# QA Chain Creation
system_prompt = "Use the following pieces of retrieved context to answer the question.\n\n{context}"
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
```

### 3. Conversational Memory with Session Tracking

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Invoking conversational RAG
response = conversational_rag_chain.invoke(
    {"input": "What is YOLO?"},
    config={"configurable": {"session_id": "User_1"}}
)

print(response["answer"])
```

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).
