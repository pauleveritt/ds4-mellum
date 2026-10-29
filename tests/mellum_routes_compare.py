"""Compare native route events while allowing independently assigned profile IDs."""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def compare_spools(reference: Path, owned: Path) -> bool:
    name = "mellum_routes_to_trace"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "tools/mellum_routes_to_trace.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    reader = sys.modules[name].SpoolReader
    with reference.open("rb") as left, owned.open("rb") as right:
        a, b = reader(left), reader(right)
        if a.cap_tokens != b.cap_tokens:
            return False
        while True:
            x, y = a.read_forward(), b.read_forward()
            if x is None or y is None:
                return x is None and y is None
            if (x.forward, x.first_position, x.rows, x.phase, x.segment_id, x.tokens, x.selected) != (
                y.forward, y.first_position, y.rows, y.phase, y.segment_id, y.tokens, y.selected
            ):
                return False


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: mellum_routes_compare.py REFERENCE OWNED")
    try:
        equal = compare_spools(Path(sys.argv[1]), Path(sys.argv[2]))
    except (OSError, ValueError) as exc:
        sys.exit(str(exc))
    if not equal:
        sys.exit("owned routes differ from reference")
