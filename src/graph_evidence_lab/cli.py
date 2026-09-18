import argparse
import json
from .data import Graph, load_queries
from .model import load_model
from .training import benchmark
from .retrieval import retrieve, context_packet


def main():
    parser = argparse.ArgumentParser(description='Train and inspect graph-aware evidence retrieval.')
    commands = parser.add_subparsers(dest='command', required=True)
    ingest=commands.add_parser('ingest',help='Import Markdown documents and their explicit local links')
    ingest.add_argument('directory');ingest.add_argument('--available-at',required=True)
    ingest.add_argument('--output',required=True)
    train = commands.add_parser('benchmark')
    train.add_argument('--graph', default='examples/engineering-graph.json')
    train.add_argument('--queries', default='examples/queries.json')
    train.add_argument('--output', default='runs/baseline')
    train.add_argument('--epochs',type=int,default=100)
    train.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    query = commands.add_parser('query')
    query.add_argument('text');query.add_argument('--graph',default='examples/engineering-graph.json')
    query.add_argument('--checkpoint',default='runs/baseline/gnn-seed-0.json')
    query.add_argument('--method',choices=['bm25','diffusion','gnn'],default='gnn')
    query.add_argument('--as-of',default='2026-09-18');query.add_argument('--context',action='store_true')
    serve = commands.add_parser('serve')
    serve.add_argument('--graph',default='examples/engineering-graph.json')
    serve.add_argument('--checkpoint',default='runs/baseline/gnn-seed-0.json')
    serve.add_argument('--port',type=int,default=8000)
    args = parser.parse_args()
    if args.command=='ingest':
        from pathlib import Path
        from .ingest import import_markdown
        graph=import_markdown(args.directory,args.available_at)
        output=Path(args.output)
        if output.exists():parser.error('Output already exists; choose a new path to preserve the prior corpus.')
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(graph.to_dict(),indent=2),encoding='utf8')
        print(json.dumps({'nodes':len(graph.nodes),'edges':len(graph.edges),'fingerprint':graph.fingerprint,'output':str(output)}))
    elif args.command=='benchmark':
        if not 1 <= args.epochs <= 1000 or 0 not in args.seeds:
            parser.error('Use 1–1000 epochs and include seed 0 for the fixed replay.')
        graph = Graph.load(args.graph)
        benchmark(graph,load_queries(args.queries,graph),args.output,args.seeds,args.epochs)
    elif args.command=='query':
        graph = Graph.load(args.graph)
        model = load_model(args.checkpoint,graph.fingerprint) if args.method=='gnn' else None
        result = retrieve(graph,args.text,args.as_of,args.method,model)
        print(json.dumps(context_packet(result) if args.context else result,indent=2))
    else:
        import uvicorn
        from .api import create_app
        uvicorn.run(create_app(args.graph,args.checkpoint),host='127.0.0.1',port=args.port)


if __name__=='__main__':main()
