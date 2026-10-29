"""Publish a complete child inside a new, exclusively reserved output container."""
from collections.abc import Callable
from pathlib import Path
import shutil
import tempfile


def publish(container: Path, build: Callable[[Path], None]) -> Path:
    """Return container/artifact after build has checked every flush and close.

    An interrupted reservation is incomplete and must never be reused. Only the
    artifact child is a published result. The parent directory must exist.
    """
    container.mkdir()  # Outside cleanup: an existing path belongs to someone else.
    stage = None
    try:
        stage = Path(tempfile.mkdtemp(prefix='.building-', dir=container))
        build(stage)
        artifact = container / 'artifact'
        stage.rename(artifact)
        return artifact
    except BaseException as original:
        try:
            if stage is not None:
                shutil.rmtree(stage)
            container.rmdir()
        except OSError as cleanup:
            original.add_note(f'publication cleanup failed: {cleanup}')
        raise
