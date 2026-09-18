"""Local FastAPI service. No URLs from evidence are fetched; no LLM is called."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from .data import Graph
from .model import load_model
from .retrieval import retrieve, context_packet


class Query(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    as_of: str = '2026-09-18'
    method: Literal['bm25','diffusion','gnn'] = 'gnn'
    k: int = Field(default=5, ge=1, le=20)


def create_app(graph_path, checkpoint_path):
    graph = Graph.load(graph_path)
    model = load_model(checkpoint_path, graph.fingerprint)
    app = FastAPI(title='Graph Evidence Lab', version='0.1.0')

    @app.get('/health')
    def health():
        return {'status':'ok','graph_fingerprint':graph.fingerprint,'nodes':len(graph.nodes)}

    @app.get('/graph')
    def graph_view(as_of: str='2026-09-18'):
        try: return graph.snapshot(as_of).to_dict()
        except ValueError as e: raise HTTPException(422,str(e)) from e

    @app.post('/retrieve')
    def rank(q: Query):
        try: return retrieve(graph,q.text,q.as_of,q.method,model if q.method=='gnn' else None,q.k)
        except ValueError as e: raise HTTPException(422,str(e)) from e

    @app.post('/context')
    def context(q: Query):
        return context_packet(rank(q))
    return app
