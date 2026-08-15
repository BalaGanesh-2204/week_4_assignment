Example Questions

After adding suitable Markdown documents, try questions such as

What is machine learning?

What is the difference between supervised and
unsupervised learning?

Explain neural networks.

What is TCP?

What is cloud computing?

What are the advantages of virtualization?

The answers should be generated only from the information contained in the ingested documents.

🛠️ Troubleshooting
Problem: No Markdown files found

Make sure your documents are inside:

docs/

and have the .md extension.

Problem: Missing API key

Check your .env file.

Make sure the required keys exist:

GROQ_API_KEY
GEMINI_API_KEY
PINECONE_API_KEY
Problem: Pinecone dimension mismatch

The Gemini embedding dimension and Pinecone index dimension must be identical.

Example:

Gemini Dimension = 768
Pinecone Dimension = 768
Problem: No useful answer

Make sure the documents were ingested before asking questions.

Run:

python ingest.py

Then restart the Streamlit application.

🔄 Complete Workflow

The complete application workflow is:

1. Add Markdown files to docs/

2. Run:
   python ingest.py

3. Documents are chunked.

4. Gemini generates embeddings.

5. Embeddings are stored in Pinecone.

6. Start Streamlit:
   streamlit run app.py

7. Enter a question.

8. Gemini generates an embedding for the question.

9. Pinecone searches for similar chunks.

10. Retrieved chunks are sent to Groq.

11. Groq generates a grounded answer.

12. Streamlit displays the answer and sources.
🔮 Future Improvements

The current implementation can be extended with:

Better Markdown-aware chunking
Sentence-based chunking
Recursive chunking
Metadata filtering
Hybrid search
Reranking
Streaming Groq responses
Conversation history
Chat history
Document upload through Streamlit
Automatic document re-indexing
Individual document deletion
RAG evaluation
Retrieval evaluation
RAGAS integration
Authentication
Multiple document collections
Token usage tracking
Response latency tracking
🎯 Learning Objectives

This project demonstrates the major components of a RAG system:

Document loading
Document chunking
Text embeddings
Vector databases
Semantic search
Context retrieval
Prompt construction
LLM generation
Source citation
Streamlit deployment
📌 Technologies

Python

Streamlit

Google Gemini

Groq

Pinecone

Python-dotenv

Markdown

Vector Search

Retrieval-Augmented Generation

👨‍💻 Author

Balaganesh

📄 License

This project is created for educational and project development purposes.