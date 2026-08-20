"""
Self-check for merge_adapters.py — this touches the final submission's
weights directly, so a silent bug here is the worst kind: it would produce a
plausible-looking adapter that scores badly for reasons nobody would trace
back to the merge step.

    python src/test_merge_adapters.py
"""

import json
import os
import shutil
import tempfile

import torch
from safetensors.torch import load_file, save_file

from merge_adapters import merge

CFG = {"r": 32, "lora_alpha": 64, "target_modules": ["q_proj", "v_proj"],
       "peft_type": "LORA", "task_type": "CAUSAL_LM"}


def make_adapter(path, value, cfg=None):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "adapter_config.json"), "w") as f:
        json.dump(cfg or CFG, f)
    save_file({"a.lora_A.weight": torch.full((4, 4), float(value)),
               "a.lora_B.weight": torch.full((4, 4), float(value) * 2)},
              os.path.join(path, "adapter_model.safetensors"))


def main():
    tmp = tempfile.mkdtemp()
    try:
        a1, a2, a3 = (os.path.join(tmp, n) for n in ("a1", "a2", "a3"))
        out = os.path.join(tmp, "merged")

        # Known values -> known mean. This is the arithmetic the whole
        # technique rests on; if this is wrong nothing downstream matters.
        make_adapter(a1, 1.0)
        make_adapter(a2, 2.0)
        make_adapter(a3, 3.0)
        merge([a1, a2, a3], out)
        merged = load_file(os.path.join(out, "adapter_model.safetensors"))
        assert torch.allclose(merged["a.lora_A.weight"], torch.full((4, 4), 2.0)), \
            "equal-weight average of 1,2,3 should be 2"
        assert torch.allclose(merged["a.lora_B.weight"], torch.full((4, 4), 4.0))
        assert os.path.exists(os.path.join(out, "adapter_config.json"))

        # Weighted average — same arithmetic, non-uniform weights.
        out2 = os.path.join(tmp, "merged_weighted")
        merge([a1, a2], out2, weights=[3.0, 1.0])
        m2 = load_file(os.path.join(out2, "adapter_model.safetensors"))
        expected = (3.0 * 1.0 + 1.0 * 2.0) / 4.0  # = 1.25
        assert torch.allclose(m2["a.lora_A.weight"], torch.full((4, 4), expected)), \
            f"weighted average wrong: got {m2['a.lora_A.weight'][0,0].item()}, want {expected}"

        # Averaging an adapter with itself must be a no-op.
        out3 = os.path.join(tmp, "merged_self")
        merge([a1, a1], out3)
        m3 = load_file(os.path.join(out3, "adapter_model.safetensors"))
        assert torch.allclose(m3["a.lora_A.weight"], torch.full((4, 4), 1.0))

        # Incompatible LoRA config must be REFUSED, not silently averaged.
        bad_cfg = dict(CFG, r=16)
        make_adapter(a3, 3.0, cfg=bad_cfg)
        try:
            merge([a1, a3], os.path.join(tmp, "should_not_exist"))
            raise AssertionError("expected SystemExit for mismatched r, got none")
        except SystemExit:
            pass
        make_adapter(a3, 3.0)  # restore for any later use

        # Same target_modules, different LIST ORDER must NOT be refused --
        # it's a set semantically, and separate training processes serialize
        # it in different orders. Regression test for a real false-positive
        # bug that blocked the first real 4-adapter merge in production.
        reordered_cfg = dict(CFG, target_modules=list(reversed(CFG["target_modules"])))
        make_adapter(a3, 3.0, cfg=reordered_cfg)
        merge([a1, a3], os.path.join(tmp, "reordered_ok"))  # must not raise
        make_adapter(a3, 3.0)  # restore for any later use

        # Incompatible tensor shapes must be REFUSED.
        bad_shape = os.path.join(tmp, "bad_shape")
        os.makedirs(bad_shape, exist_ok=True)
        with open(os.path.join(bad_shape, "adapter_config.json"), "w") as f:
            json.dump(CFG, f)
        save_file({"a.lora_A.weight": torch.ones(8, 8),
                   "a.lora_B.weight": torch.ones(8, 8)},
                  os.path.join(bad_shape, "adapter_model.safetensors"))
        try:
            merge([a1, bad_shape], os.path.join(tmp, "should_not_exist2"))
            raise AssertionError("expected SystemExit for mismatched shape, got none")
        except SystemExit:
            pass

        print("all checks passed")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
