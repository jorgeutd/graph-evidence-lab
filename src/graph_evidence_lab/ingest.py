"""Import local Markdown and explicit local links; never infer edges from labels."""
from pathlib import Path
from urllib.parse import urlsplit, unquote
import hashlib
import os
import re
from .data import Graph, day


def import_markdown(directory, available_at):
    day(available_at)
    root=Path(directory).resolve(strict=True)
    if not root.is_dir():raise ValueError('Input must be a directory')
    files=[]
    for folder,dirs,names in os.walk(root,followlinks=False):
        dirs[:]=sorted(d for d in dirs if not d.startswith('.') and d not in {'node_modules','__pycache__'} and not (Path(folder)/d).is_symlink())
        for name in sorted(names):
            path=Path(folder)/name
            if path.suffix.lower()=='.md' and not path.is_symlink():
                resolved=path.resolve()
                if root not in resolved.parents:raise ValueError('Input escaped the selected directory')
                files.append(resolved)
                if len(files)>500:raise ValueError('Import at most 500 Markdown documents per experiment')
    if not files:raise ValueError('No Markdown documents found')
    nodes=[];ids={};texts={}
    for path in sorted(files):
        if path.stat().st_size>100_000:raise ValueError('Split large documents before importing: '+path.name)
        text=path.read_text(encoding='utf8').strip()
        if not text or len(text)>20_000:raise ValueError('Each document must contain 1–20,000 characters: '+path.name)
        relative=path.relative_to(root).as_posix()
        slug=re.sub('[^a-z0-9]+','-',path.stem.lower()).strip('-') or 'document'
        identity=slug[:40]+'-'+hashlib.sha256(relative.encode()).hexdigest()[:10]
        match=re.search(r'^#\s+(.+)$',text,re.MULTILINE)
        title=match.group(1).strip() if match else path.stem
        ids[path]=identity;texts[path]=text
        nodes.append(dict(id=identity,title=title,text=text,kind='document',source='markdown://'+relative,available_at=available_at))
    pairs=set()
    for path,text in texts.items():
        for destination in re.findall(r'\[[^\]\n]*\]\(([^\s)]+)(?:\s+"[^"\n]*")?\)',text):
            parsed=urlsplit(destination.strip('<>'))
            if parsed.scheme or parsed.netloc or not parsed.path or parsed.path.startswith('/'):
                continue
            target=(path.parent/unquote(parsed.path)).resolve()
            if target in ids and target!=path:pairs.add((ids[path],ids[target]))
    edges=[dict(source=a,target=b,relation='references',confidence=1.0,available_at=available_at) for a,b in sorted(pairs)]
    return Graph({'nodes':nodes,'edges':edges})
