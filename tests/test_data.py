import copy
import json
import pytest
from graph_evidence_lab.data import Graph, load_queries
from graph_evidence_lab.features import edge_tensors
from graph_evidence_lab.retrieval import relations

def test_temporal_exclusion_including_reverse_edges(graph):
    snap = graph.snapshot('2026-09-18')
    assert len(snap.nodes) == 36
    assert all('future-runtime' not in (e.source,e.target) for e in snap.edges)
    edges,_,_ = edge_tensors(snap,relations(graph))
    assert edges.max() < len(snap.nodes)
    assert len(graph.snapshot('2027-01-01').nodes) == 37

def test_future_edge_between_present_nodes_excluded(graph):
    data = graph.to_dict(); data['edges'][0]['available_at']='2027-01-01'
    assert len(Graph(data).snapshot('2026-09-18').edges) == len(graph.snapshot('2026-09-18').edges)-1

@pytest.mark.parametrize('kind',['duplicate_node','duplicate_edge','missing_endpoint','confidence','date'])
def test_bad_graph_rejected(graph,kind):
    data=graph.to_dict()
    if kind=='duplicate_node': data['nodes'].append(data['nodes'][0])
    if kind=='duplicate_edge': data['edges'].append(data['edges'][0])
    if kind=='missing_endpoint': data['edges'][0]['target']='unknown'
    if kind=='confidence': data['edges'][0]['confidence']=float('nan')
    if kind=='date': data['nodes'][0]['available_at']='yesterday'
    with pytest.raises(ValueError): Graph(data)

def test_fingerprint_order_invariant(graph):
    data=graph.to_dict(); data['nodes'].reverse(); data['edges'].reverse()
    assert Graph(data).fingerprint==graph.fingerprint

@pytest.mark.parametrize('kind',['group','future_label','empty_grade'])
def test_bad_split_rejected(graph,queries,tmp_path,kind):
    if kind=='group': queries[-1]['group']=queries[0]['group']
    if kind=='future_label': queries[-1]['relevant']={'future-runtime':3}
    if kind=='empty_grade': queries[-1]['relevant']={}
    path=tmp_path/'queries.json';path.write_text(json.dumps(queries))
    with pytest.raises(ValueError):load_queries(path,graph)
