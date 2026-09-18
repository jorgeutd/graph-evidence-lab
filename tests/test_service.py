import pytest
from fastapi.testclient import TestClient
from graph_evidence_lab.api import create_app
from graph_evidence_lab.evaluation import metrics
from graph_evidence_lab.model import EvidenceRanker, save_model
from graph_evidence_lab.retrieval import retrieve,relations,context_packet,citation_check
from conftest import ROOT

def test_metrics_hand_calculation():
    values=metrics(['b','x','a'],{'a':3,'b':1},k=3)
    assert values['recall_at_k']==1
    assert values['mrr_at_k']==1
    assert values['ndcg_at_k']==pytest.approx((1+7/2)/(7+1/1.584962500721156))

def test_no_lexical_anchor_abstains(graph):
    r=retrieve(graph,'zzqqxxvv','2026-09-18','bm25')
    assert r['status']=='insufficient_evidence' and not r['results']
    assert context_packet(r)['status']=='insufficient_evidence'

def test_context_budget_and_citation_membership(graph):
    r=retrieve(graph,'graph evidence','2026-09-18','bm25')
    packet=context_packet(r,600)
    assert packet['characters']==len(packet['context'])<=600
    assert packet['included_ids']
    assert citation_check('Answer ['+packet['included_ids'][0]+']',packet)['valid_references']
    assert not citation_check('Answer [invented]',packet)['valid_references']
    assert not citation_check('No citations',packet)['valid_references']

@pytest.fixture
def client(graph,tmp_path):
    path=tmp_path/'model.json';save_model(EvidenceRanker(relations(graph)),path,graph.fingerprint)
    return TestClient(create_app(ROOT/'examples/engineering-graph.json',path))

def test_api_evidence_and_context(client):
    assert client.get('/health').json()['nodes']==37
    assert len(client.get('/graph').json()['nodes'])==36
    request={'text':'cache memory','method':'bm25'}
    result=client.post('/retrieve',json=request)
    assert result.status_code==200 and result.json()['trace']['excluded_future_nodes']==1
    assert client.post('/context',json=request).json()['status']=='ready'

@pytest.mark.parametrize('request',[{'text':''},{'text':'x','as_of':'bad'},{'text':'x','k':21},{'text':'x','method':'unknown'},{'text':' '*5}])
def test_api_invalid_request(client,request):
    assert client.post('/retrieve',json=request).status_code==422
