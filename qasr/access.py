"""Fail-fast check that the Hugging Face token can read every gated repo."""
from __future__ import annotations

from .config import Config
from .utils import get_logger

log = get_logger("qasr.access")


def required_repos(cfg: Config) -> list[tuple[str, str]]:
    repos: list[tuple[str, str]] = []
    for s in cfg.data.sources:
        repos.append((s.repo, "dataset"))
    lv = cfg.live
    repos += [(cfg.features.whisper_model, "model"), (lv.mt_hi_en, "model"), (lv.mt_en_hi, "model"),
              (cfg.models.indicconformer if lv.hindi_asr == "indicconformer" else cfg.models.indicwav2vec, "model")]
    if lv.english_asr_backend == "transformers":
        repos.append((lv.english_asr, "model"))
    seen, out = set(), []
    for r in repos:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def check_access(cfg: Config) -> bool:
    try:
        from huggingface_hub import auth_check, whoami
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    except ImportError:
        log.warning("huggingface_hub too old for auth_check; skipping access check.")
        return True
    try:
        user = whoami().get("name")
        log.info("Hugging Face login OK: %s", user)
    except Exception:
        log.error("Not logged in to Hugging Face. Run:  hf auth login   (paste a READ token)")
        return False
    ok = True
    for repo, kind in required_repos(cfg):
        try:
            auth_check(repo, repo_type=kind)
            log.info("  [ok]      %s", repo)
        except GatedRepoError:
            ok = False
            url = f"https://huggingface.co/{'datasets/' if kind == 'dataset' else ''}{repo}"
            log.error("  [GATED]   %s  -> open %s and click 'Agree and access repository'", repo, url)
        except RepositoryNotFoundError:
            ok = False
            log.error("  [MISSING] %s (repo not found or no permission)", repo)
        except Exception as e:  # network etc.
            log.warning("  [?]       %s (%s)", repo, e)
    return ok
