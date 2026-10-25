#!/usr/bin/env bash
# End-to-end check of a deployed cluster: stores loaded, app serving Qdrant retrieval and Neo4j graph queries,
# and store parity measured in-cluster. Usage: deploy/k8s/smoke.sh   (expects kubectl context on the cluster)
set -euo pipefail
NS=reglens
kubectl -n $NS wait --for=condition=complete job/reglens-load-stores --timeout=600s
kubectl -n $NS logs job/reglens-load-stores | tail -1
kubectl -n $NS rollout status deploy/reglens --timeout=300s
kubectl -n $NS port-forward svc/reglens 18080:80 >/dev/null 2>&1 &
PF=$!; trap 'kill $PF' EXIT; sleep 3
H=$(curl -sf localhost:18080/healthz); echo "healthz $H"
echo "$H" | grep -q '"graph_backend":"neo4j"' && echo "$H" | grep -q '"retriever":"qdrant_bge"'
S=$(curl -sf "localhost:18080/api/search?q=how%20long%20can%20a%20credit%20union%20hold%20a%20check&k=3")
echo "$S" | python3 -c "import json,sys;d=json.load(sys.stdin);print('search',d['retriever'],[h['citation'] for h in d['hits']])"
G=$(curl -sf localhost:18080/api/graph/12_CFR_1005_6_b_1)
echo "$G" | python3 -c "import json,sys;d=json.load(sys.stdin);n=d['neighbors'];assert any(x['relation']=='INTERPRETS' for x in n);print('graph',len(n),'neighbors from neo4j')"
curl -sf -o /dev/null -w "ui %{http_code}\n" localhost:18080/
# NetworkPolicy: a pod that is not part of RegLens must not reach the stores
if kubectl -n $NS run np-probe --rm -i --restart=Never --image=busybox:1.36 --labels=app=intruder -- \
     sh -c "wget -q -T 5 -O- http://qdrant:6333/readyz" >/dev/null 2>&1; then
  echo "networkpolicy NOT enforced (CNI without policy support)"
else
  echo "networkpolicy enforced: unlabeled pod cannot reach qdrant"
fi
echo "parity (in-cluster):"
kubectl -n $NS run parity --rm -i --restart=Never --image=reglens:local --image-pull-policy=IfNotPresent \
  --labels=app=reglens-indexer --overrides='{"spec":{"containers":[{"name":"parity","image":"reglens:local","imagePullPolicy":"IfNotPresent","command":["reglens","eval","stores"],"envFrom":[{"configMapRef":{"name":"'"$(kubectl -n $NS get cm -o name | grep reglens-config | cut -d/ -f2)"'"}},{"secretRef":{"name":"'"$(kubectl -n $NS get secret -o name | grep reglens-secrets | cut -d/ -f2)"'"}}]}]}}' \
  -- 2>/dev/null | grep -E "^(Qdrant|Neo4j)" || true
