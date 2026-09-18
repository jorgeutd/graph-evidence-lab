from pathlib import Path
import pytest
from graph_evidence_lab.data import Graph, load_queries

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def graph():
    return Graph.load(ROOT/'examples/engineering-graph.json')

@pytest.fixture
def queries(graph):
    return load_queries(ROOT/'examples/queries.json', graph)
