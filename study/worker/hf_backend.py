"""HF/accelerate generation backend for the study worker.

The alternative to the vLLM backend, and the one that works on next5/next6.

Why this exists. Qwen3-8B in bf16 (~16.4GB) does not fit one 16GB a4000, and
vLLM's answer — `--tensor-parallel-size 2` — is unusable on those nodes because
NCCL's cross-rank collectives hang there (see README "GPU sizing"). But the
repo has been generating with the 8B on next6 all along via a different
mechanism: HF + accelerate `device_map="auto"`, which splits the model by
LAYER and dispatches the forward pass sequentially across cards. That moves
activations with plain tensor copies and issues NO NCCL collectives, so it is
unaffected by the broken interconnect.

This backend is that path, wrapped in the same interface the vLLM backend
exposes. It is a deliberate mirror of `eval/eval_endorsement_heldout.py`:
same `load_llama` loader, same `build_chat_input` chat-template call, same
`system=None` bare-user-turn shape, and the same out-of-range-token retry.
That eval is where our adapter's reported numbers come from, so matching it
is what makes the study's `ours` arm the same model behaviour we measured.

Base vs adapter. One PeftModel serves every arm: `ours` runs with the
adapter active, and the base arms run inside `model.disable_adapter()`, which
is PEFT's exact analogue of addressing the base model name on a vLLM engine.
No second model is loaded and no weights are swapped per request.
"""

import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_STUDY_DIR = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_STUDY_DIR)

sys.path.insert(0, _STUDY_DIR)
sys.path.insert(0, os.path.join(_REPO_ROOT, "training_mitigations", "core"))

import config
from llama_local import build_chat_input, load_llama
from vllm_server import validate_adapter


class NonFiniteLogitsGuard:
    """Logits processor that keeps corrupted logits out of the sampler.

    Why this is needed. The multi-GPU `device_map=auto` path occasionally
    produces non-finite logits mid-generation — the same class of flaky
    placement/offload numerics event that `eval_endorsement_heldout
    .generate_one` guards against on the token-id side. When it happens,
    `torch.multinomial` trips a CUDA device-side assert.

    A device-side assert is unrecoverable: it poisons the whole CUDA context,
    so every later call in the process fails instantly and an in-process retry
    is impossible. (That is exactly what happened on the first HF smoke run —
    one bad generation took the worker down for every arm.) So instead of
    recovering after the fact, this replaces non-finite entries BEFORE the
    sampler sees them, keeping the context clean.

    It does NOT silently pass off a sanitized generation as a good one: it sets
    `self.fired`, and the caller discards that generation and regenerates. The
    substituted values only exist to let the forward pass finish without
    killing the process.
    """

    def __init__(self):
        self.fired = False

    def reset(self):
        self.fired = False

    def __call__(self, input_ids, scores):
        bad = ~torch.isfinite(scores)
        if bool(bad.any()):
            self.fired = True
            # -1e4 rather than -inf: an all -inf row would make softmax produce
            # NaN and trip the same assert we are trying to avoid.
            scores = scores.masked_fill(bad, -1e4)
        return scores


class HFBackend:
    """Holds the loaded policy and generates one stage at a time.

    Not thread-safe and deliberately serial: the study issues one generation
    per participant, so there is nothing to batch, and a single in-flight
    forward keeps peak memory predictable on 16GB cards.
    """

    def __init__(self, device_map="auto", device="cuda", max_memory=None):
        validate_adapter(config.LORA_PATH, config.BASE_MODEL_ID)
        self.model, self.tokenizer = load_llama(
            model_id=config.BASE_MODEL_ID,
            device=device,
            device_map=device_map,
            max_memory=max_memory,
        )
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, config.LORA_PATH)
        self.model.eval()
        self.logits_guard = NonFiniteLogitsGuard()
        # How many generations were discarded as corrupted. Worth watching: a
        # rate above a few percent means the placement is unhealthy, not flaky.
        self.corruption_events = 0

    def describe(self):
        """One-line summary of where the weights actually landed."""
        devices = sorted({str(p.device) for p in self.model.parameters()})
        return "%s + LoRA %s on %s (HF device_map)" % (
            config.BASE_MODEL_ID, os.path.basename(config.LORA_PATH), ",".join(devices))

    def model_label(self, adapter):
        """Name recorded in the job's per-stage data, mirroring the vLLM
        backend's served-model-name so the two are comparable in analysis."""
        return config.LORA_NAME if adapter == "lora" else config.BASE_MODEL_ID

    def generate(self, adapter, system, user_text):
        """Generate one stage's output.

        `adapter` is the generation plan's selector: "lora" runs with our
        adapter active, None runs the frozen base. Returns stripped text and
        raises on an empty generation, matching the vLLM backend's contract.
        """
        assert adapter in (None, "lora"), "unknown adapter selector: %r" % (adapter,)
        if adapter is None:
            with self.model.disable_adapter():
                return self._generate(system, user_text)
        return self._generate(system, user_text)

    def _generate(self, system, user_text):
        """Sample once through the same chat-template path training used.

        Two flaky-numerics guards, both specific to the multi-GPU
        `device_map=auto` path and both retried ONCE because the underlying
        event does not recur:

          - non-finite logits, caught before the sampler by
            NonFiniteLogitsGuard (see its docstring for why prevention rather
            than recovery);
          - out-of-range token ids, carried over verbatim from
            eval_endorsement_heldout.generate_one, which crash the decode.

        A second occurrence of either is systematic (bad checkpoint, broken
        placement) rather than flaky, so it raises and the worker reports a
        failed attempt.
        """
        device = next(self.model.parameters()).device
        input_ids = build_chat_input(self.tokenizer, user_text, system=system, device=device)
        in_len = input_ids.shape[1]
        gen_kwargs = dict(
            max_new_tokens=int(config.GEN_MAX_TOKENS),
            num_return_sequences=1,
            do_sample=True,
            temperature=float(config.GEN_TEMPERATURE),
            top_p=float(config.GEN_TOP_P),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            logits_processor=[self.logits_guard],
            # Frees the worker from a pathologically slow generation; see
            # config.GEN_MAX_SECONDS. transformers stops cleanly and returns
            # what it has, which we then reject below rather than show.
            max_time=float(config.GEN_MAX_SECONDS),
        )
        vocab_n = len(self.tokenizer)
        for attempt in range(2):
            self.logits_guard.reset()
            started = time.time()
            with torch.no_grad():
                output_ids = self.model.generate(input_ids, **gen_kwargs)
            elapsed = time.time() - started
            new_ids = output_ids[0, in_len:]
            # `max_time` truncates mid-sentence rather than erroring, so a
            # generation that ran out the clock without emitting EOS is
            # incomplete and must not be shown to a participant.
            timed_out = (elapsed >= config.GEN_MAX_SECONDS * 0.98
                         and int(new_ids[-1]) != self.tokenizer.eos_token_id)
            assert not timed_out, (
                "generation hit the %ds wall-clock cap after %d tokens without "
                "finishing; the node is too contended to serve this arm"
                % (config.GEN_MAX_SECONDS, new_ids.shape[0]))
            corrupted = self.logits_guard.fired
            corrupted = corrupted or bool(((new_ids < 0) | (new_ids >= vocab_n)).any())
            if not corrupted:
                text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                assert text, "HF backend produced empty text"
                return text
            self.corruption_events += 1
            assert attempt == 0, (
                "generation was corrupted twice in a row (non-finite logits or "
                "out-of-range token ids); this is systematic, not flaky"
            )
        raise AssertionError("unreachable")
