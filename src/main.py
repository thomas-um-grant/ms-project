# Import necessary libraries
from database import relational_api

result = relational_api.get_corpus_path(
    dataset_name="Dataset 1",
    doc_id="Doc 1",
    corpus_id=1,
)

print(result)

result2 = relational_api.get_corpuses_by_dataset(dataset_name="Dataset 1")
print(f"Num of corpuses: {len(result2)}")

for corpus in result2:
    print(corpus.keys())
    print(corpus["corpus_path"])

# Fetch and print the SQLite version

# ChromaDB
## Get ChromaDB connection details
## Initialize ChromaDB client

# API
## Endpoints
### Health Check

### Add Document

### Search Documents
