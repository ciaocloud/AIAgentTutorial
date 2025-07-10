# Tutorial: Retrieval Augmented Generation (RAG)

Welcome to the tutorial on Retrieval Augmented Generation (RAG). This guide will explain what RAG is, why it's useful, and how to build a simple RAG pipeline using Python and the Gemini API.

## What is RAG?

Retrieval Augmented Generation is a technique that enhances the accuracy and reliability of Large Language Models (LLMs) by grounding them with information from an external knowledge source.

LLMs are trained on vast amounts of data, but their knowledge is frozen at a specific point in time. They can also "hallucinate" or make up facts. RAG helps to solve these problems by:

-  **Retrieving** relevant information from your own documents or a specific knowledge base.
-  **Augmenting** the user's prompt with this retrieved information.
-  **Generating** an answer based on the provided context.

This ensures the LLM's response is based on factual, up-to-date, and contextually relevant information.

## The RAG Architecture

A typical RAG system has three main components. Our implementation now uses a more sophisticated semantic search approach.

1.  **Indexing/Knowledge Base**: This is the where we prepare the information that the system searches through. The source of the knowledge base could include document collections, databases, web pages, APIs, streams, etc. 
We take the texts, split them into chunks, and then use an **embedding model** (such as `BERT`) to convert each chunk into a numerical vector. These vectors are usually stored and indexed in **vector databases** for efficient search later in the query stage.
2.  **Retrieval System**: When a user asks a question, the retrieval system would apply certain mechanism to find the relevant information. This could be, but is not limited to, vector **similarity search** using embeddings, in which the system first converts the query into its own vector embedding, it then may use cosine similarity to compare the query's vector with all the stored document vectors from the knowledge base. The chunks with the highest similarity (i.e., the ones that are most conceptually related) are selected. This process may involve setting retrieval parameters (e.g., top-k, threshold), multi-step re-ranking, etc. Also, some other retrieval mechanisms such as keyword-based search (BM25, TF-IDF) can be used.
3.  **Generator/LLM**: The retrieved context chunks are then combined with the original query and fed to a generative LLM. The LLM is instructed to generate an answer *only* using the information provided in the context.

### What are Embeddings?

At its core, an embedding is a way of representing text (or other data like images and audio) as a list of numbers, called a **vector**. This numerical representation captures the semantic meaning and context of the original text.

Think of it like a sophisticated coordinate system for words and sentences. Words with similar meanings will have vectors that are "close" to each other in this coordinate space. For example, the vectors for "cat" and "kitten" would be very close, while the vector for "skyscraper" would be far away.

This is incredibly powerful because it allows a computer to understand the relationships between concepts without needing to understand the language itself. In our RAG pipeline, we use this to find document chunks that are conceptually related to the user's query, not just those that share the same keywords.

#### How to Choose an Embedding Model

When building a RAG system, the choice of embedding model is crucial. Here are a few factors to consider:

1.  **Performance**: How well does the model capture the meaning of your text? This is often measured by its performance on benchmark datasets. For general-purpose tasks, models like Gemini's `text-embedding-004` are a great starting point.
2.  **Dimensionality**: This refers to the size of the vector the model produces (e.g., 768 or 1536 numbers). Higher dimensions can capture more detail but require more storage and computation. For most applications, a model with 768 dimensions is a good balance.
3.  **Cost & Speed**: Embedding text costs money and takes time. Consider the pricing and latency of the model, especially if you are working with a large number of documents or need real-time performance.
4.  **Domain-Specific Needs**: If you are working in a highly specialized domain (e.g., legal documents, biomedical research), you might need a model that has been specifically trained on text from that field to achieve the best results.

<!-- For this tutorial, we use `text-embedding-004`, which is a high-quality, general-purpose model suitable for a wide range of tasks. -->

#### Popular Embedding Model Options

Here are a few popular and effective embedding models available today:

**OpenAI Embeddings**
- Strong performance across domains, pay-per-use API model (~$0.02/1M tokens)
- **Popular Models**: `text-embedding-3-large` (1536 dim), `text-embedding-3-small` (1024 dim)
- **Use Cases**: General-purpose applications, high-accuracy requirements
- **Example**: 
  ```python
  ## assume we have OpenAI client set up using API key
  response = client.embeddings.create(
      model="text-embedding-3-large",
      input="Your text here"
  )
  embedding = response.data[0].embedding
  ```

**Open Source Sentence-BERT (SBERT)**
- Open-source (via libraries like Sentence-Transformers). Fast, good general performance. Run locally.
- **Popular Models**: `all-MiniLM-L6-v2` (384 dim), `all-mpnet-base-v2` (768 dim). Please refer to https://sbert.net/docs/sentence_transformer/pretrained_models.html for a list of pretrained embedding model.
- **Example**:
  ```python
  # !uv add sentence_transformers
  from sentence_transformers import SentenceTransformer
  embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
  embeddings = embedding_model.encode(["Your text here"])
  ```

**E5 (Microsoft)**
- Strong performance, multiple sizes available. Free through Hugging Face, requires compute.
- **Models**: `e5-small` (384 dim), `e5-base` (768 dim), `e5-large` (1024 dim)
- **Example**:
  ```python
  from transformers import AutoTokenizer, AutoModel
  tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-large-v2')
  model = AutoModel.from_pretrained('intfloat/e5-large-v2')
  ```

**Cohere Embeddings**
- **Strengths**: Excellent multilingual support, good performance. API-based pricing, free tier available.
- **Models**: `embed-english-v3.0`, `embed-multilingual-v3.0`
- **Dimensions**: 1024
- **Example**:
  ```python
  import cohere
  co = cohere.Client(COHERE_API_KEY)
  response = co.embed(texts=["Your text here"], model='embed-english-v3.0')
  embedding = response.embeddings[0]
  ```

**Google Gemini**:
- `text-embedding-004` (768 dim): Google's latest, high-performance model.
- Other Gemini models also offer embedding capabilities, such as `embedding-001`, `gemini-embedding-exp-03-07`, `gemini-embedding-exp`.
- **Example**:
  ```python
  client = genai.Client(api_key=GOOGLE_API_KEY)
  response = client.models.embed_content(
    model="models/text-embedding-004",
    contents=["Your text here"],
  )
  embeddings = [e.values for e in response.embeddings]
  ```



## A Hands-On Example

### 1. Build Knowledge Base (Document Store)

We first use an LLM to make up two short stories, one for Garfield, and the other for Doraemon. Our goal is to successfully retrieve relevant information specific from these stories.

```python
text = """
Garfield, a plump orange tabby, spent most of his days in a blissful state of napping. He loved undoubtedly
  eating, especially lasagna, which he considered the pinnacle of culinary achievement. When not devouring food or sleeping,
  Garfield enjoyed tormenting his owner, Jon Arbuckle, and kicking Odie, the cheerful beagle, off the table. Jon often tried to get Garfield to exercise, but the cat preferred to watch TV, another one of his cherished pastimes. Despite his lazy demeanor, Garfield had a sharp wit and a deep philosophical appreciation for Mondays (or rather, his disdain for them).
Doraemon, the blue robotic cat from the future, found Nobita crying again. "Suneo and Gian took my new comic book!" Nobita
  wailed. Doraemon sighed, then pulled out the "Anywhere Door" from his four-dimensional pocket. "Let's go get it back," he
  said, and they stepped through, appearing instantly in Gian's room. Gian, startled, quickly returned the comic, and Nobita
  cheered, forgetting his tears.
"""
```
To build our knowledge base, we first split the text into chunks. For this example, we simply split by sentences. In a real-world application, you probably need to experiment with different chunk sizes (100-1000 tokens) based on your content type and use case, and use overlapping chunks (10-20% overlap) to ensure important information isn't split across boundaries.
```python
import re

chunks = re.split(r'(?<=[.!?])\s+', text) ## split by ., !, or ? followed by a space
chunks = [s.strip() for s in chunks if s.strip()] ## filter out empty strings
```

Next, we take the string of each chunk and build a numeric vector by using the embedding model. For this example, we will use Gemini's `text-embedding-004` model. This step is central to converting our text data into a format that can be used for similarity comparisons.

```python
embedding_model = "models/text-embedding-004"
response = client.models.embed_content(model=embedding_model, content=chunks)
embeddings = [e.values for e in response.embeddings]
```
Now we obtained 10 vectors in a list, each has 768 dimensions. These embeddings would typically be stored in a vector database for efficient retrieval. Popular choices of vector database include:
- Chroma: Simple embedded vector database
- FAISS: Facebook's similarity search library
- Qdrant: High-performance vector database
- Pinecone: Managed vector database service
- Weaviate: Open-source vector database

We use `ChromaDB` here.
```python
#! uv add chromadb
import chromadb

DB_NAME = "my_story_knowledge_base"

db_client = chromadb.Client()
db_collection = db_client.get_or_create_collection(name=DB_NAME)
db_collection.add(
    embeddings=embeddings,
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)
# print(db_collection.peek(1))

```

### 2. Retrieval of Relevant Documents

Now we would like to ask the question "what are Garfield's favorate hobbies?" and get the answer from the given story.
We need to retrieve the relevant information from it, or more specifically, from the vector database we just wrote into. To do that, we first convert our query into an embedding, and then calculates the cosine similarity between the query embedding and all document embeddings. It returns the `top_k` most similar document chunks. Using Chroma database's API, we can simply call its `query` method. 
```python
query = "what are Garfield's favorate hobbies?"
response = client.models.embed_content(
    model=embedding_model,
    contents=[query],
    config=types.EmbedContentConfig(
        task_type="retrieval_query",
    )
)
query_embeddings  = [e.values for e in response.embeddings]

results = db_collection.query(
    query_embeddings=query_embeddings, 
    n_results=3)
print(results["documents"])
```
Note that you will switch to the `retrieval_query` mode of embedding generation. Although not mandatory, it's best practice to use `task_type='RETRIEVAL_DOCUMENT'` when embedding your document chunks and `task_type='RETRIEVAL_QUERY'` when embedding user queries. This ensures that both the documents and queries are embedded in a way that maximizes their compatibility and the accuracy of the retrieval process.

### 3. Generate Final Response

<!-- The `generate_answer` function acts as our generator. It takes the original query and the retrieved context, constructs a prompt, and sends it to the `gemini-pro` LLM. The LLM is instructed to answer *only* based on the provided context. -->

Finally, we can assemble the retrieved context information with the original query in a prompt, and ask the LLM to perform the **augmented generation** for a final answer, using both the query and retrieved context. 


```python
context = "\n".join(results["documents"][0])
prompt = f"""
Answer the user's query based ONLY on the provided context. If the context does not contain the answer, say 'The provided context does not have the answer to this question'.

    Context:
    {context}

    Query: {query}

    Answer:
"""
model="gemini-2.5-flash"
response = client.models.generate_content(
    model=model,
    contents=prompt,
)
print(response.text)
```

Here is the answer provided by Gemini, pretty awesome, right?
```
Garfield's favorite hobbies include watching TV, tormenting Jon Arbuckle, kicking Odie off the table, napping, and devouring food.
```

## TODO: Re-ranking

## TODO: Limitations

## Next Steps

Congratulations! You've now built a RAG pipeline.

