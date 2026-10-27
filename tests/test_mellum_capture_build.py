"""Dry-run build contracts; never initializes a GPU or executes a model."""
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


def dry_run(*args: str) -> str:
    result = subprocess.run(["make", "-n", *args], cwd=ROOT, capture_output=True, text=True, check=True)
    return result.stdout


def test_default_build_still_builds_all_products() -> None:
    assert dry_run() == dry_run("all"), "default make stopped selecting all"


def test_backend_object_override_retains_native_capture_dependencies() -> None:
    commands = dry_run("-B", "ds4", "CORE_OBJS=ds4.o")
    links = [line for line in commands.splitlines() if " -o ds4 " in line]
    assert links, "dry-run did not contain the product link"
    for name in ("routes", "profile", "capture_state", "observer"):
        assert f"ds4_mellum_{name}.o" in links[-1], f"backend override dropped {name}"


def test_standalone_metal_consumers_link_observer_dependencies() -> None:
    targets = (
        "tests/test_mellum_kernels", "tests/test_qwen4_kernels", "tests/test_glm53_kda",
        "tests/test_glm_attention", "tests/test_mxfp4_metal",
    )
    commands = dry_run("-B", "UNAME_S=Darwin", *targets)
    for target in targets:
        links = [line for line in commands.splitlines() if f" -o {target} " in line]
        assert links, f"dry-run did not contain the {target} link"
        for name in ("observer", "profile"):
            assert f"ds4_mellum_{name}.o" in links[-1], f"{target} dropped {name}"


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in sorted(globals().items())
                              if name.startswith("test_") and callable(value))


if __name__ == "__main__":
    unittest.main()
