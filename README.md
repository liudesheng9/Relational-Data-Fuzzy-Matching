# Relational Data Fuzzy Matching
Fast fuzzy matching for all-kinds of string data.

## Highlights
- 🚀 a fast data pipeline for string relational data fuzzy matching.
- ⚡️ 720X faster than 48 thread fuzzywuzzy fuzzy matching in 35,000 X 84,000,000 fuzzy matching task.
- 💾 Disk-space efficient, store large matrix as .npz format sparce matrix.
- 🖥️ Supports macOS, Linux, Windows, and any other operate system can run Python.

## Documentation
This is a **bare bone implementation** of the algorithm. You can change the data structure in the various part of matching pipeline to fit the different concrete problems.

The basic pipeline can be divided in to these parts:

INPUT --> PREPROCESS --> TF-IDF VECTORIZATION --> COSINE SIMILARITY CALCULATION --> 1ST FILTERING --> BUILD MATCHTABLE --> 2ND FILTERING --> OUTPUT
