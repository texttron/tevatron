import os


def ensure_parent_dir(path):
    """Create parent directories for a filesystem path (``mkdir -p`` semantics).

    Intended for output *file* paths: ensures the containing directory exists
    before long-running encode/search/rerank work so runs fail fast on
    permission errors instead of after compute.

    No-op when *path* is falsy or resolves to having no parent (e.g.
    ``output.pkl`` in the current directory).
    """
    if not path:
        return
    parent = os.path.dirname(os.path.normpath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
