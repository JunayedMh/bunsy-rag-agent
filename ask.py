import os
from ragkit import RAG

rag = RAG(
    document = r"C:\Users\Junayed\Downloads\Bunsy Bloggers Brief.pdf",
    model = "openrouter/free",
    base_url="https://openrouter.ai/api/v1",
)

response = rag.query("What is this document about?")
print(response.answer)
print("\nSources:")
for source in response.sources:
    print(f"- {source.document} (page {source.page})")