"""Host-only checks for the C2 compile-target UB capability boundary."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

UTILS = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/triton_utils.py"


def load_utils(ub_result):
    calls = {"soc": [], "query": 0}
    torch = types.ModuleType("torch")
    torch.npu = types.SimpleNamespace(
        current_device=lambda: 0,
        get_device_name=lambda _index: "Ascend910B3",
    )

    vllm = types.ModuleType("vllm")
    triton_utils = types.ModuleType("vllm.triton_utils")
    triton_utils.HAS_TRITON = True
    triton_utils.tl = types.SimpleNamespace(insert_slice=object(), extract_slice=object(), get_element=object())
    driver_utils = types.SimpleNamespace(get_device_properties=lambda _index: {"num_aicore": 20, "num_vectorcore": 40})
    triton_utils.triton = types.SimpleNamespace(
        runtime=types.SimpleNamespace(driver=types.SimpleNamespace(active=types.SimpleNamespace(utils=driver_utils)))
    )
    vllm.triton_utils = triton_utils

    ascend = types.ModuleType("vllm_ascend")
    ascend.__path__ = []
    envs = types.ModuleType("vllm_ascend.envs")
    envs.VLLM_ASCEND_ROPE_UB_SIZE_KB = 0
    ascend.envs = envs

    platform = types.ModuleType("tbe.common.platform")

    def set_soc(soc):
        calls["soc"].append(soc)

    def get_spec(key):
        calls["query"] += 1
        if isinstance(ub_result, Exception):
            raise ub_result
        if key != "UB_SIZE":
            raise AssertionError(key)
        return ub_result

    platform.set_current_compile_soc_info = set_soc
    platform.get_soc_spec = get_spec
    tbe = types.ModuleType("tbe")
    tbe.__path__ = []
    common = types.ModuleType("tbe.common")
    common.__path__ = []
    tbe.common = common
    common.platform = platform

    replacements = {
        "torch": torch,
        "vllm": vllm,
        "vllm.triton_utils": triton_utils,
        "vllm_ascend": ascend,
        "vllm_ascend.envs": envs,
        "tbe": tbe,
        "tbe.common": common,
        "tbe.common.platform": platform,
    }
    saved = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)
    try:
        spec = importlib.util.spec_from_file_location("pr2_c2_utils_test", UTILS)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:
        restore(saved)
        raise
    return module, calls, saved


def restore(saved):
    for name, previous in saved.items():
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


class C2CapabilityTests(unittest.TestCase):
    def test_compiler_ub_is_cached_separately_from_runtime_fallback(self):
        module, calls, saved = load_utils(196608)
        try:
            self.assertIsNone(module.try_get_compile_target_ub_bytes())
            module.init_device_properties_triton()
            module.init_device_properties_triton()
            self.assertEqual(module.get_vectorcore_num(), 40)
            self.assertEqual(module.try_get_compile_target_ub_bytes(), 196608)
            self.assertEqual(calls, {"soc": ["Ascend910B3"], "query": 1})
        finally:
            restore(saved)

    def test_unknown_compiler_ub_does_not_borrow_runtime_fallback(self):
        for result in (RuntimeError("no TBE capability"), True, "196608"):
            module, calls, saved = load_utils(result)
            try:
                module.init_device_properties_triton()
                module.init_device_properties_triton()
                self.assertEqual(module.get_vectorcore_num(), 40)
                self.assertEqual(module.get_ub_size_bytes(), 196608)
                self.assertIsNone(module.try_get_compile_target_ub_bytes())
                self.assertEqual(calls["query"], 1)
            finally:
                restore(saved)


if __name__ == "__main__":
    unittest.main()
