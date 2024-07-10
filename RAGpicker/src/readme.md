# RAGEval
A comprehensive python package to perform gridsearch based optimization for you RAG usecase

##### Current capability can search all possible combinations of -
- LLM Model
- Embedding Model
- chunk_size
- chunk_overlap
- K (top matching chunks)

### Usage
Refer to `src/runnable_script.py` for an example run.

`cd` to `/src` folder and execute `python runnable_script.py`.

Currently, any code that uses this package will have to reside within the `/src/` (or should import relative to the directory) folder since this library isn't available as a packaged module.