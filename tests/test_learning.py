import copy
import torch
import pytest
from graph_evidence_lab.features import bm25, diffuse
from graph_evidence_lab.model import EvidenceRanker, RelationalLayer, save_model, load_model
from graph_evidence_lab.retrieval import relations, tensors
from graph_evidence_lab.training import fit

def test_lexical_exact_match():
    scores=bm25('cache memory',['cache memory cache','unrelated text','memory allocation'])
    assert scores[0]>scores[2]>scores[1]

def test_diffusion_conserves_probability_with_dangling_node():
    values=diffuse(torch.tensor([2.,1.,3.]),torch.tensor([[0,1],[1,0]]),torch.ones(2))
    assert torch.all(values>=0) and values.sum().item()==pytest.approx(1,abs=1e-6)

def test_message_aggregation_known_weights():
    layer=RelationalLayer(2,1)
    with torch.no_grad():
        layer.message_projection.weight.copy_(torch.eye(2));layer.relation.weight.zero_()
    _,detail=layer(torch.tensor([[2.,4.],[6.,8.],[0.,0.]]),torch.tensor([[0,1],[2,2]]),torch.tensor([0,0]),torch.tensor([1.,.5]))
    assert torch.allclose(detail['aggregate'][2],torch.tensor([5/3,8/3]))
    assert detail['denominator'][2]==1.5

def test_permutation_equivariance(graph):
    torch.manual_seed(3);g=graph.snapshot('2026-09-18');model=EvidenceRanker(relations(g)).eval()
    x,_,edge=tensors(g,'cache memory',model)
    p=torch.randperm(len(x)); inverse=torch.argsort(p)
    original=model(x,*edge)
    changed=model(x[p],inverse[edge[0]],edge[1],edge[2])
    assert torch.allclose(changed,original[p],atol=1e-6)

def test_isolated_nodes_and_zero_weight_edges_are_finite(graph):
    model=EvidenceRanker(relations(graph));x,_,edge=tensors(graph.snapshot('2026-09-18'),'graph',model)
    scores=model(x,edge[0],edge[1],torch.zeros_like(edge[2]))
    empty=model(x,edge[0][:,:0],edge[1][:0],edge[2][:0])
    assert torch.allclose(scores,empty)
    scores.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)

def test_edges_change_computation(graph):
    torch.manual_seed(0);model=EvidenceRanker(relations(graph));x,_,e=tensors(graph,'graph',model)
    assert not torch.allclose(model(x,*e),model(x,e[0][:,:0],e[1][:0],e[2][:0]))

def test_checkpoint_roundtrip_and_graph_guard(graph,tmp_path):
    model=EvidenceRanker(relations(graph));path=tmp_path/'weights.json';save_model(model,path,graph.fingerprint)
    loaded=load_model(path,graph.fingerprint)
    assert all(torch.equal(v,loaded.state_dict()[k]) for k,v in model.state_dict().items())
    with pytest.raises(ValueError,match='fingerprint'):load_model(path,'different')

def test_test_labels_cannot_change_training(graph,queries):
    changed=copy.deepcopy(queries)
    for q in changed:
        if q['split']=='test':q['relevant']={'batching':100}
    first,record=fit(graph,queries,seed=4,epochs=2)
    second,other=fit(graph,changed,seed=4,epochs=2)
    assert record==other
    assert all(torch.equal(v,second.state_dict()[k]) for k,v in first.state_dict().items())
