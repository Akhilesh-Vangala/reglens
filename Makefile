.PHONY: install build test index eval-retrieval eval-answers
install:
	python -m pip install -e ".[dev]"
build:
	reglens build
test:
	pytest -q
index:
	reglens index
eval-retrieval:
	reglens eval retrieval --retrievers bm25 dense hybrid azure_hybrid
eval-answers:
	reglens eval answers --configs A_closed_book B_rag_no_verify C_rag_verified
