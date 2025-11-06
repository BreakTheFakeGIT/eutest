import langchain 
import plotly 
import pandas
import bs4

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)

# Sample text
text = "LangChain is a powerful framework for building applications with LLMs. It provides tools for chaining, memory, agents, and more."

# 1. RecursiveCharacterTextSplitter
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    separators=["\n\n", "\n", ".", " ", ""]
)
recursive_chunks = recursive_splitter.split_text(text)

# 2. CharacterTextSplitter
char_splitter = CharacterTextSplitter(
    separator=" ",
    chunk_size=40,
    chunk_overlap=5
)
char_chunks = char_splitter.split_text(text)

# 3. TokenTextSplitter (requires tiktoken or similar tokenizer)
token_splitter = TokenTextSplitter(
    chunk_size=20,
    chunk_overlap=5
)
token_chunks = token_splitter.split_text(text)

# Print results
print("RecursiveCharacterTextSplitter Chunks:")
for chunk in recursive_chunks:
    print("-", chunk)

print("\nCharacterTextSplitter Chunks:")
for chunk in char_chunks:
    print("-", chunk)

print("\nTokenTextSplitter Chunks:")
for chunk in token_chunks:
    print("-", chunk)


splitters = {
    "recursive": RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50),
    "character": CharacterTextSplitter(separator=" ", chunk_size=300, chunk_overlap=50),
    "token": TokenTextSplitter(chunk_size=300, chunk_overlap=50)
}

text = "Your document or input text here"
results = {name: splitter.split_text(text) for name, splitter in splitters.items()}

for name, chunks in results.items():
    print(f"\n{name.upper()} Splitter:")
    for chunk in chunks:
        print("-", chunk)



from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)
import pandas as pd
import plotly.express as px

# Sample text
sample_text = (
    "LangChain is a powerful framework for building applications with large language models (LLMs). "
    "It provides tools for chaining, memory, agents, and more. "
    "Text splitting is a crucial step in preparing documents for embedding or LLM processing. "
    "Different splitters offer different strategies for chunking text based on characters, tokens, or structure."
)

# Define splitters
splitters = {
    "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10),
    "CharacterTextSplitter": CharacterTextSplitter(separator=" ", chunk_size=50, chunk_overlap=10),
    "TokenTextSplitter": TokenTextSplitter(chunk_size=50, chunk_overlap=10)
}

# Compute metrics
def compute_metrics(name, chunks):
    num_chunks = len(chunks)
    avg_length = sum(len(chunk) for chunk in chunks) / num_chunks
    overlap_ratio = sum(1 for i in range(1, num_chunks) if chunks[i-1][-10:] in chunks[i]) / (num_chunks - 1) if num_chunks > 1 else 0
    return {
        "Splitter": name,
        "Num Chunks": num_chunks,
        "Avg Chunk Length": avg_length,
        "Overlap Ratio": overlap_ratio
    }

# Run benchmark
results = []
for name, splitter in splitters.items():
    chunks = splitter.split_text(sample_text)
    results.append(compute_metrics(name, chunks))

# Create DataFrame
df = pd.DataFrame(results)
print(df)

# Plot results
fig = px.bar(df, x="Splitter", y=["Num Chunks", "Avg Chunk Length", "Overlap Ratio"],
             barmode="group", title="Chunk Quality Benchmark")
fig.show()


from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Sample HTML input
html_text = """<html><body>
<h1>Header 1</h1>
<p>This is a paragraph.</p>
<div class="exclude">This block should be excluded.</div>
<b>Bold text here.</b>
<h2>Header 2</h2>
<p>Another paragraph with forbidden content.</p>
</body></html>"""

# Parse HTML
soup = BeautifulSoup(html_text,

# Define tags to include and exclusion rules
include_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'b', 'strong']
exclude_classes = ['exclude']
exclude_phrases = ['forbidden content', 'do not include']

# Extract relevant text
included_texts = []
for tag in soup.find_all(include_tags):
    # Exclude by class
    if any(cls in exclude_classes for cls in tag.get('class', [])):
        continue

    text = tag.get_text(strip=True)

    # Exclude by phrase
    if any(phrase.lower() in text.lower() for phrase in exclude_phrases):
        continue

    included_texts.append(text)

# Combine extracted text
combined_text = "\n".join(included_texts)

# Split into chunks using LangChain
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)
chunks = text_splitter.split_text(combined_text)

# Output chunks
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n")

