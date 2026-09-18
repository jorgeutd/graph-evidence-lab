import pytest
from graph_evidence_lab.ingest import import_markdown

def test_markdown_links_are_explicit_local_and_reproducible(tmp_path):
    (tmp_path/'a.md').write_text('# Alpha\nCache memory. [Next](b.md#details) [Again](b.md) [Web](https://example.com/b.md) [Escape](../private.md)')
    (tmp_path/'b.md').write_text('# Beta\nA second document.')
    graph=import_markdown(tmp_path,'2026-09-18')
    assert len(graph.nodes)==2 and len(graph.edges)==1
    assert graph.edges[0].relation=='references'
    assert graph.fingerprint==import_markdown(tmp_path,'2026-09-18').fingerprint
    assert all(str(tmp_path) not in n.source for n in graph.nodes)

def test_hidden_directories_and_symlinks_are_not_ingested(tmp_path):
    (tmp_path/'a.md').write_text('# Alpha\nText')
    (tmp_path/'.hidden').mkdir();(tmp_path/'.hidden'/'b.md').write_text('# Hidden\nText')
    (tmp_path/'alias.md').symlink_to(tmp_path/'a.md')
    assert len(import_markdown(tmp_path,'2026-09-18').nodes)==1

@pytest.mark.parametrize('text',['','x'*20001])
def test_invalid_document_requires_explicit_preparation(tmp_path,text):
    (tmp_path/'a.md').write_text(text)
    with pytest.raises(ValueError):import_markdown(tmp_path,'2026-09-18')
