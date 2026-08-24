# Retrieval-Augmented Generation (RAG)

## 1. What is RAG?

RAG stands for **Retrieval-Augmented Generation**.

It is a technique where an application first retrieves relevant information from a knowledge source and then provides that information to an LLM to generate the final response.

In simple terms:

> RAG = Retrieve relevant information + Give it to the LLM + Generate an answer

Basic flow:

```text
Documents
    ↓
Retrieve relevant information
    ↓
LLM
    ↓
Final Answer
```

RAG is useful when the information required to answer a question is stored in external sources such as company documents, PDFs, databases, FAQs, or knowledge bases.

---

## 2. Why is RAG Needed?

An LLM can generate answers based on the information it already knows and the information provided in the current prompt.

However, an application may have its own private or frequently changing information.

For example:

- Company policies
- Product documentation
- Customer FAQs
- Internal documentation
- Medical information
- Legal documents
- Technical documentation

An LLM does not automatically know the contents of these private documents.

Instead of trying to train the LLM again with every document, RAG allows the application to retrieve the relevant information and provide it to the LLM at the time of the question.

### Without RAG

```text
User Question
      ↓
     LLM
      ↓
   Answer
```

### With RAG

```text
User Question
      ↓
Retrieve Relevant Information
      ↓
Provide Retrieved Information to LLM
      ↓
   Answer
```

---

# 3. Main Components of RAG

A basic RAG system contains several important components.

```text
Documents
    ↓
Document Chunks
    ↓
Embedding Model
    ↓
Vector Database
    ↓
Similarity Search
    ↓
Retrieved Context
    ↓
LLM
    ↓
Final Answer
```

---

## 3.1 Documents

Documents are the original sources containing the information that the application wants to make available to the LLM.

Examples:

- PDF files
- Text files
- Word documents
- Websites
- Company documentation
- FAQs
- Database records

Example:

```text
appointment_policy.pdf
doctor_guidelines.pdf
faq.txt
```

These documents become the knowledge source for the RAG system.

---

## 3.2 Document Chunking

Large documents are usually divided into smaller pieces called **chunks**.

For example:

```text
Large Document
      ↓
┌───────────────┐
│   Chunk 1     │
├───────────────┤
│   Chunk 2     │
├───────────────┤
│   Chunk 3     │
├───────────────┤
│   Chunk 4     │
├───────────────┤
│      ...      │
└───────────────┘
```

Why do we split documents?

A user question usually requires only a small part of a large document.

For example, if a document contains an entire appointment policy and the user asks:

> "Can I cancel an appointment two hours before?"

We want to retrieve the section related to cancellation rather than the entire document.

Chunking makes it easier to find the relevant information.

---

## 3.3 Embedding Model

An embedding model converts text into a numerical vector.

For example:

```text
"Patients can cancel appointments..."
                ↓
        Embedding Model
                ↓
[0.12, -0.32, 0.51, ...]
```

The resulting vector represents the semantic meaning of the text.

Similar pieces of text tend to have similar vectors.

For example:

```text
"How can I cancel my appointment?"

and

"Can a patient cancel a scheduled appointment?"
```

have different words but similar meanings.

Their embeddings should therefore be relatively close in vector space.

In the 03_embedding.py practical, `nomic-embed-text` with Ollama was used to experiment with embeddings.

---

## 3.4 Vector Database

The embeddings generated from document chunks need to be stored somewhere so they can be searched efficiently.

A **vector database** or vector index stores these vectors and allows similarity searches.

Examples include:

- FAISS
- Chroma
- Pinecone
- Qdrant
- Weaviate

A simplified representation is:

```text
Vector Database

┌────────────────────────────────────┐
│ Vector            Document Chunk   │
├────────────────────────────────────┤
│ [0.12, ...]  →    Chunk 1          │
│ [0.43, ...]  →    Chunk 2          │
│ [0.87, ...]  →    Chunk 3          │
│ [0.21, ...]  →    Chunk 4          │
└────────────────────────────────────┘
```

In our 03_embedding.py experiment, FAISS was used for basic vector similarity search.

---

## 3.5 User Query

When a user asks a question, that question also needs to be converted into an embedding.

For example:

```text
User Question

"Can I cancel my appointment two hours before?"
                ↓
        Embedding Model
                ↓
          Query Vector
```

This is similar to the `embed_query()` experiment performed earlier.

---

## 3.6 Similarity Search / Retrieval

The query vector is compared against the vectors stored in the vector database.

The system finds the chunks that are most similar to the user's question.

```text
Query Vector
     ↓
Vector Database
     ↓
Similarity Search
     ↓
Most Relevant Chunks
```

Example:

```text
Question:

"Can I cancel my appointment two hours before?"

              ↓

        Similarity Search

              ↓

┌─────────────────────────────────┐
│ Chunk 17                         │
│ Cancellation within 2 hours...  │
├─────────────────────────────────┤
│ Chunk 21                         │
│ Cancellation refund policy...   │
└─────────────────────────────────┘
```

This process is called **retrieval**.

---

## 3.7 Retrieved Context

The relevant chunks retrieved from the vector database are provided to the LLM as context.

For example:

```text
Retrieved Context:

"Patients can cancel appointments up to
two hours before the scheduled time."
```

The LLM can now use this information when generating its response.

---

## 3.8 LLM

The LLM receives:

1. Instructions
2. Retrieved context
3. User question

Conceptually:

```text
Instructions
     +
Retrieved Context
     +
User Question
     ↓
    LLM
     ↓
Final Answer
```

The LLM uses the retrieved information to generate a natural-language response.

---

# 4. Complete RAG Pipeline

The complete RAG pipeline can be represented as:

```text
                         INDEXING PHASE

┌──────────────────┐
│    Documents     │
│ PDF / TXT / DB   │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Document Chunks  │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Embedding Model  │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Vector Database  │
│   FAISS/Chroma   │
└──────────────────┘


                         QUERY PHASE

┌──────────────────┐
│   User Question  │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Query Embedding  │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Similarity Search│
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Relevant Chunks  │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Prompt + Context │
│ + User Question  │
└────────┬─────────┘
         ↓
┌──────────────────┐
│       LLM        │
└────────┬─────────┘
         ↓
┌──────────────────┐
│   Final Answer   │
└──────────────────┘
```

---

# 5. Information Flow

The information flows through the RAG system in two major phases.

## Phase 1 — Indexing

This phase prepares the documents before users ask questions.

```text
Documents
    ↓
Chunking
    ↓
Embedding
    ↓
Vector Database
```

For example:

```text
appointment_policy.pdf
        ↓
     Chunks
        ↓
 Embedding Model
        ↓
    Vectors
        ↓
   Vector Store
```

The vector database is now ready to search.

---

## Phase 2 — Query and Retrieval

When the user asks a question:

```text
User Question
      ↓
Query Embedding
      ↓
Vector Similarity Search
      ↓
Relevant Document Chunks
      ↓
Prompt + Context
      ↓
LLM
      ↓
Final Answer
```

---

# 6. Simple Example

Imagine an application has this document:

```text
Appointment Policy

Patients can cancel appointments up to two hours
before the scheduled appointment time.

Cancellations made less than two hours before the
appointment may not be eligible for a refund.
```

The document is processed:

```text
Document
   ↓
Chunks
   ↓
Embeddings
   ↓
Vector Database
```

Now the user asks:

> "Can I cancel my appointment two hours before?"

The question is converted into an embedding:

```text
User Question
      ↓
Query Embedding
      ↓
Query Vector
```

The vector database searches for similar vectors.

It finds the relevant chunk:

```text
"Patients can cancel appointments up to two hours
before the scheduled appointment time."
```

That information is provided to the LLM.

The LLM generates:

```text
Yes. Patients can cancel appointments up to two
hours before the scheduled appointment time.
```

This is a simple example of RAG.

---

# 7. RAG vs Normal LLM

## Normal LLM

```text
┌───────────────┐
│ User Question │
└───────┬───────┘
        ↓
┌───────────────┐
│      LLM      │
└───────┬───────┘
        ↓
┌───────────────┐
│     Answer    │
└───────────────┘
```

## RAG

```text
┌───────────────┐
│ User Question │
└───────┬───────┘
        ↓
┌────────────────────┐
│ Query Embedding    │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Vector Search      │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Relevant Context   │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Prompt + Context   │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│        LLM         │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│    Final Answer    │
└────────────────────┘
```

The main difference is that RAG retrieves external information before asking the LLM to generate the answer.

---

# 8. RAG Does Not Mean Training the LLM

A common misunderstanding is that RAG means training or fine-tuning the LLM using your documents.

That is not what basic RAG does.

Instead:

```text
Your Documents
      ↓
Embeddings
      ↓
Vector Database
```

The LLM remains separate.

When a question arrives:

```text
User Question
      +
Retrieved Information
      ↓
     LLM
      ↓
Final Answer
```

This means the knowledge source can be updated without retraining the LLM.

---

# 9. Relationship Between Embeddings, FAISS, and RAG

The concepts explored in the Week-3 practical are connected.

### Embeddings

Convert text into vectors.

```text
Text
 ↓
Embedding Model
 ↓
Vector
```

### FAISS

Searches vectors for similarity.

```text
Query Vector
 ↓
FAISS
 ↓
Similar Vectors
```

### RAG

Combines retrieval with LLM generation.

```text
Documents
 ↓
Embeddings
 ↓
FAISS
 ↓
Relevant Documents
 ↓
LLM
 ↓
Answer
```

Therefore:

> Embeddings and vector search are important building blocks of a RAG system.

---

# 10. Key Takeaways

- RAG stands for **Retrieval-Augmented Generation**.
- RAG allows an LLM to use information from external knowledge sources.
- Documents are usually divided into smaller chunks.
- Embedding models convert chunks into numerical vectors.
- Vector databases store and search those vectors.
- A user's question is also converted into a vector.
- Similarity search retrieves relevant document chunks.
- Retrieved chunks are provided to the LLM as context.
- The LLM uses the context and user question to generate the final answer.
- FAISS and Chroma can be used for vector similarity search.
- RAG is different from training or fine-tuning an LLM.

---

# 11. Overall Mental Model

The simplest way to remember RAG is:

```text
             KNOWLEDGE
                 │
                 ↓
          ┌─────────────┐
          │  Documents  │
          └──────┬──────┘
                 ↓
            Embeddings
                 ↓
          Vector Database
                 │
                 │ Search
                 ↑
                 │
          User Question
                 ↓
        Relevant Information
                 ↓
        Prompt + Context
                 ↓
                LLM
                 ↓
            Final Answer
```

> **RAG = Find the relevant information first, then let the LLM use that information to answer the user.**
