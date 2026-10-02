"""
Model loading and tagging helpers for run_deidentify.py. Top-level imports are
standard library only; the deidentify packages are imported inside the
functions that need them.
"""

import gc
import platform
from pathlib import Path

DEFAULT_MODEL = "model_bilstmcrf_ons_large-v0.2.0"

# Synthetic, invented text used only by --smoke-test. No real person or record.
SMOKE_TEXT = (
    "Patient J. de Vries (67 jaar) werd op 12-03-2024 gezien in het "
    "Radboudumc te Nijmegen. Verslag doorgebeld aan dr. Jansen, "
    "tel 06-12345678."
)


def patch_old_torch_pickles():
    """Let torch 1.12/1.13 load the 2020 deidentify Flair models.

    The released models were pickled with an older torch. Several flair 0.10
    embedding classes (PooledFlairEmbeddings among them) restore themselves
    with `self.__dict__ = d` and never call torch's own Module.__setstate__,
    which is what normally adds the hook registries newer torch versions
    expect. load_state_dict then fails with
        AttributeError: 'PooledFlairEmbeddings' object has no attribute
        '_load_state_dict_post_hooks'
    This creates any such registry empty on first access, which is exactly
    what Module.__setstate__ would have done. Weights are not touched.
    """
    from collections import OrderedDict

    import torch

    factories = {
        "_load_state_dict_post_hooks": OrderedDict,
        "_load_state_dict_pre_hooks": OrderedDict,
        "_state_dict_hooks": OrderedDict,
        "_state_dict_pre_hooks": OrderedDict,
        "_forward_pre_hooks": OrderedDict,
        "_forward_hooks": OrderedDict,
        "_backward_hooks": OrderedDict,
        "_non_persistent_buffers_set": set,
        "_is_full_backward_hook": lambda: None,
        # registries added in torch 2.x
        "_backward_pre_hooks": OrderedDict,
        "_forward_hooks_with_kwargs": OrderedDict,
        "_forward_pre_hooks_with_kwargs": OrderedDict,
        "_forward_hooks_always_called": OrderedDict,
    }
    original = torch.nn.Module.__getattr__
    if getattr(original, "_ghmschrft_patched", False):
        return

    def __getattr__(self, name):
        factory = factories.get(name)
        if factory is not None and name not in self.__dict__:
            value = factory()
            self.__dict__[name] = value
            return value
        return original(self, name)

    __getattr__._ghmschrft_patched = True
    torch.nn.Module.__getattr__ = __getattr__


def build_tagger(model: str, mini_batch_size: int):
    """Return a deidentify tagger for a model name or a path to a model file."""
    patch_old_torch_pickles()
    from deidentify.taggers import CRFTagger, FlairTagger
    from deidentify.tokenizer import TokenizerFactory

    tokenizer = TokenizerFactory().tokenizer(corpus="ons", disable=("tagger", "ner"))
    name = Path(model).name
    if name.startswith("model_crf_") or name.endswith(".pickle"):
        return CRFTagger(model=model, tokenizer=tokenizer)
    return FlairTagger(model=model, tokenizer=tokenizer,
                       mini_batch_size=mini_batch_size, verbose=False)


def package_versions() -> dict:
    versions = {"python": platform.python_version()}
    for mod in ("deidentify", "flair", "torch", "spacy", "deduce"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 -- version reporting must never fail the run
            versions[mod] = "unknown"
    return versions


class PooledMemory:
    """Snapshot and restore the memory of every PooledFlairEmbeddings module."""

    def __init__(self, tagger):
        self.modules = []
        import torch

        flair_model = getattr(tagger, "tagger", None)
        if isinstance(flair_model, torch.nn.Module):  # CRF models have no pooled memory
            from flair.embeddings import PooledFlairEmbeddings
            self.modules = [m for m in flair_model.modules()
                            if isinstance(m, PooledFlairEmbeddings)]
        # The dict values are replaced, never modified in place (torch.min/max/add
        # return new tensors), so a shallow copy of each dict is a full snapshot.
        self.snapshots = [(dict(m.word_embeddings), dict(m.word_count)) for m in self.modules]

    def words_in_memory(self):
        return [len(m.word_embeddings) for m in self.modules]

    def restore(self):
        for m, (emb, cnt) in zip(self.modules, self.snapshots):
            m.word_embeddings = dict(emb)
            m.word_count = dict(cnt)


def is_oom(err: Exception) -> bool:
    msg = str(err).lower()
    return "not enough memory" in msg or "out of memory" in msg


def _empty_cuda_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 -- best effort only
        pass


def device_name() -> str:
    try:
        import flair
        import torch
        if str(flair.device).startswith("cuda"):
            return f"{flair.device} ({torch.cuda.get_device_name(0)})"
        return str(flair.device)
    except Exception:  # noqa: BLE001
        return "unknown"


def make_annotate(tagger, pool: PooledMemory, pool_mode: str, stats: dict):
    from deidentify.base import Document

    def run(text):
        doc = tagger.annotate([Document(name="doc", text=text)])[0]
        return [[a.start, a.end, a.tag] for a in doc.annotations]

    def annotate(text: str):
        if not text or text.isspace():
            return []
        if pool_mode == "reset":
            pool.restore()
        try:
            return run(text)
        except RuntimeError as err:
            if not is_oom(err) or not hasattr(tagger, "mini_batch_size"):
                raise
            # Retry this one report sentence by sentence. Pooling then only sees
            # earlier sentences of the report, so its labels can differ slightly
            # from a batched run; the meta file records how often this happened.
            stats["oom_retries"] += 1
            gc.collect()
            _empty_cuda_cache()
            saved = tagger.mini_batch_size
            tagger.mini_batch_size = 1
            try:
                if pool_mode == "reset":
                    pool.restore()
                return run(text)
            finally:
                tagger.mini_batch_size = saved
                gc.collect()

    return annotate
