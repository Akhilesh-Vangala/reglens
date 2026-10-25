.PHONY: install test lint build stores eval-retrieval eval-stores eval-answers eval-copilot serve compose kind
install:
	python -m pip install -e ".[dev,serve]"
lint:
	ruff check src tests api scripts && ruff format --check src tests api scripts
test:
	pytest -q
build:            ## fetch the pinned eCFR text, parse, chunk
	reglens build
stores:           ## load the citation graph into Neo4j and vectors into Qdrant
	reglens load-stores
eval-retrieval:
	reglens eval retrieval --retrievers bm25 dense_bge hybrid_bge dense_bge+rerank graph_bge_s1
eval-stores:
	reglens eval stores
eval-answers:     ## needs a model key; prints the upper bound first
	reglens eval answers --dry-run --repeats 3 && reglens eval answers --repeats 3
eval-copilot:
	reglens eval copilot --dry-run && reglens eval copilot
serve:
	reglens-serve
compose:
	docker compose up --build
kind:             ## local Kubernetes: cluster, image, manifests, smoke test
	kind create cluster --name reglens || true
	docker build -t reglens:local . && kind load docker-image reglens:local --name reglens
	kubectl apply -k deploy/k8s && deploy/k8s/smoke.sh
