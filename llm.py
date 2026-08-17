"""
llm.py — Wrapper tipis di atas Anthropic SDK.

Kenapa dibungkus, nggak langsung panggil SDK di tiap layer?
1. Token accounting terpusat -> data mentah untuk H3 (context cost).
2. Kalau lo mau ganti provider (OpenAI/Gemini), cukup ganti file ini.
3. Retry & JSON parsing nggak perlu ditulis ulang di 4 tempat.

CATATAN BILLING: subscription Claude Pro/Max TIDAK termasuk akses API.
Bikin akun terpisah di console.anthropic.com, isi prepaid credits,
lalu simpan key-nya di environment variable ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import anthropic

from config import MODEL, get_api_key


# ----------------------------------------------------------
# Token ledger
# ----------------------------------------------------------
@dataclass
class TokenLedger:
    """
    Implementasi formula Cost(s) = |s.d| + |s.b| + Σ|r| (Gao et al., 2026).

    input_tokens  -> |s.d| + |s.b|  (dokumen + buffer riwayat)
    output_tokens -> Σ|r|           (akumulasi respons)

    Objek ini di-share ke semua layer dalam satu sesi supaya biaya
    critique & scoring IKUT terhitung. Ini penting untuk kejujuran
    metodologis: VERIFIED memanggil model >1x per pertanyaan, jadi
    jangan cuma catat call terakhir.
    """
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    by_stage: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def record(self, stage: str, in_tok: int, out_tok: int) -> None:
        self.input_tokens += in_tok
        self.output_tokens += out_tok
        self.calls += 1
        s = self.by_stage.setdefault(stage, {"in": 0, "out": 0, "calls": 0})
        s["in"] += in_tok
        s["out"] += out_tok
        s["calls"] += 1

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def cost_usd(self) -> float:
        return (
            self.input_tokens / 1_000_000 * MODEL.price_per_mtok_input
            + self.output_tokens / 1_000_000 * MODEL.price_per_mtok_output
        )

    def snapshot(self) -> Dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total,
            "llm_calls": self.calls,
            "cost_usd": round(self.cost_usd(), 6),
            "by_stage": self.by_stage,
        }

    def reset(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self.by_stage = {}


# ----------------------------------------------------------
# Client
# ----------------------------------------------------------
class LLMClient:
    """
    Satu-satunya titik kontak ke Anthropic API.

    Ganti provider? Cukup tulis ulang method .complete() supaya tetap
    mengembalikan (text, input_tokens, output_tokens). Sisa pipeline
    nggak perlu disentuh sama sekali.
    """

    def __init__(self, api_key: Optional[str] = None, ledger: Optional[TokenLedger] = None):
        key = get_api_key("ANTHROPIC_API_KEY", api_key or "")
        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY kosong. Ambil di console.anthropic.com "
                "(akun API terpisah dari subscription Claude), lalu salin "
                ".env.example jadi .env dan isi key-nya, atau:\n"
                "  export ANTHROPIC_API_KEY='sk-ant-...'"
            )
        self.client = anthropic.Anthropic(api_key=key)
        self.ledger = ledger or TokenLedger()

    def complete(
        self,
        prompt: str,
        system: str = "",
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stage: str = "generation",
        max_retries: int = 3,
    ) -> str:
        """Satu panggilan ke model. Token otomatis masuk ledger."""
        model = model or MODEL.generator_model
        max_tokens = max_tokens or MODEL.max_tokens
        temperature = MODEL.temperature if temperature is None else temperature

        last_err: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                kwargs: Dict[str, Any] = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if system:
                    kwargs["system"] = system

                resp = self.client.messages.create(**kwargs)

                self.ledger.record(
                    stage,
                    resp.usage.input_tokens,
                    resp.usage.output_tokens,
                )
                return "".join(
                    block.text for block in resp.content if block.type == "text"
                )

            except (anthropic.RateLimitError, anthropic.APIStatusError) as e:
                last_err = e
                # exponential backoff: 2s, 4s, 8s
                time.sleep(2 ** (attempt + 1))
            except Exception as e:  # noqa: BLE001
                last_err = e
                break

        raise RuntimeError(f"LLM call gagal setelah {max_retries} percobaan: {last_err}")

    def complete_json(
        self,
        prompt: str,
        system: str = "",
        fallback: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Minta output JSON dan parse dengan aman.

        LLM kadang tetap membungkus JSON dalam ```json ... ``` walau
        sudah dilarang. Jadi jangan pernah langsung json.loads() —
        selalu bersihkan dulu. Fallback mencegah 1 parse error
        merusak run 30 emiten yang sudah jalan 2 jam.
        """
        system = (system + "\n\nJawab HANYA dengan JSON valid. "
                           "Tanpa penjelasan, tanpa markdown, tanpa ```.").strip()
        raw = self.complete(prompt, system=system, **kwargs)
        return safe_json_parse(raw, fallback)

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Hitung token persis (tidak menambah biaya generation)."""
        try:
            r = self.client.messages.count_tokens(
                model=model or MODEL.generator_model,
                messages=[{"role": "user", "content": text}],
            )
            return r.input_tokens
        except Exception:  # noqa: BLE001
            from config import CONTEXT
            return int(len(text) / CONTEXT.chars_per_token)


def safe_json_parse(raw: str, fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Ekstrak objek JSON pertama dari teks bebas."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return fallback if fallback is not None else {"_parse_error": True, "_raw": raw[:500]}
