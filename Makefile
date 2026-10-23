.PHONY: install build test index eval-retrieval eval-retrieval-azure eval-answers serve
install:
	python -m pip install -e ".[dev,local]"
build:
	reglens build
test:
	pytest -q
index:
	reglens index
eval-retrieval:
	reglens eval retrieval --retrievers bm25 dense_bge hybrid_bge dense_bge+rerank hybrid_bge+rerank
eval-retrieval-azure:
	reglens eval retrieval --retrievers dense hybrid azure_hybrid
serve:
	reglens-serve
eval-answers:
	reglens eval answers --configs A_closed_book B_rag_no_verify C_rag_verified
