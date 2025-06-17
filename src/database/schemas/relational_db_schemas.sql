-- corpus table
CREATE TABLE IF NOT EXISTS corpus (
    dataset_name VARCHAR(50) NOT NULL,
    doc_id VARCHAR(50) NOT NULL,
    corpus_id INTEGER NOT NULL,
    corpus_path TEXT NOT NULL,
    PRIMARY KEY (dataset_name, doc_id, corpus_id)
);

-- queries table
CREATE TABLE IF NOT EXISTS queries (
    dataset_name VARCHAR(50) NOT NULL,
    query_id INTEGER NOT NULL,
    query TEXT NOT NULL,
    PRIMARY KEY (dataset_name, query_id)
);

-- qrels table
CREATE TABLE IF NOT EXISTS qrels (
    dataset_name VARCHAR(50) NOT NULL,
    query_id INTEGER NOT NULL,
    corpus_id INTEGER NOT NULL,
    relevance INTEGER NOT NULL,
    answer TEXT,
    PRIMARY KEY (dataset_name, query_id, corpus_id),
    FOREIGN KEY (corpus_id) REFERENCES corpus(corpus_id),
    FOREIGN KEY (query_id) REFERENCES queries(query_id)
);

-- documents table
CREATE TABLE IF NOT EXISTS documents (
    dataset_name VARCHAR(50) NOT NULL,
    doc_id VARCHAR(50) NOT NULL,
    corpus_id INTEGER NOT NULL,
    corpus_path TEXT NOT NULL,
    title VARCHAR(50),
    summary TEXT,
    company VARCHAR(50),
    industry VARCHAR(200),
    date DATE,
    location VARCHAR(100),
    PRIMARY KEY (dataset_name, doc_id, corpus_id)
);
