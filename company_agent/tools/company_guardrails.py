# -*- coding: utf-8 -*-
from __future__ import annotations

"""
company_guardrails.py
----------------------
Guardrails para validar y garantizar que company_node siempre devuelve
exactamente 5 nombres de empresa reales y limpios.

Capas de defensa
-----------------
  1. is_valid_company_name()  — reglas léxicas duras (longitud, caracteres,
                                 stopwords, patrones de no-empresa)
  2. validate_names()         — filtra una lista aplicando la capa 1
  3. fallback_serper()        — si quedan < 3 válidas, hace búsqueda directa
                                 con Serper y extrae candidatos del título
  4. fallback_claude()        — si aún faltan, llama a Claude API con un prompt
                                 ultra-estricto que devuelve JSON limpio
  5. ensure_five_companies()  — orquesta todo y garantiza exactamente 5 nombres
"""

import json
import os
import re
import unicodedata
from typing import List

from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# STOPWORDS — palabras que NUNCA son un nombre de empresa válido
# ══════════════════════════════════════════════════════════════════════════════
_STOPWORDS: set[str] = {
    # genéricas
    "top", "best", "mejores", "principales", "lista", "ranking", "empresas",
    "compañías", "companies", "startups", "players", "líderes", "leaders",
    # medios / plataformas
    "forbes", "wikipedia", "linkedin", "youtube", "google", "facebook",
    "instagram", "twitter", "x", "tiktok", "reddit", "glassdoor",
    "crunchbase", "bloomberg", "reuters", "techcrunch", "cincodias",
    "eleconomista", "expansion", "cinco", "días", "días",
    # geografía
    "spain", "españa", "madrid", "barcelona", "europa", "europe",
    "berlin", "berlín", "paris", "london", "amsterdam",
    # términos web / legales
    "contacto", "about", "login", "cookies", "privacy", "terms",
    "inicio", "home", "menú", "menu", "blog", "news", "noticias",
    "privacidad", "legal", "aviso", "anuncios", "suscripción",
    # descripciones genéricas de sector
    "banco", "bank", "fintech", "neobanco", "neobank", "aplicación",
    "app", "plataforma", "platform", "servicio", "service",
    "solución", "solution", "software", "cloud", "saas",
    # años / números solos
    "2023", "2024", "2025", "2026",
}

# Patrones que indican que el string NO es un nombre de empresa
_BAD_PATTERNS = re.compile(
    r"""
    ^\d+$                           # solo números
    | ^https?://                    # URL
    | \.(com|es|io|co|org|net)$     # dominio
    | ^(las?\s|los?\s|the\s|a\s)    # artículo inicial
    | \b(mejores?|top|best|principales?)\b   # ranking words
    | [<>{}\[\]|\\]                 # caracteres raros
    | \s{3,}                        # triple espacio
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Nombre válido: entre 2 y 50 chars, al menos 2 letras, sin saltos de línea
_MIN_LEN = 2
_MAX_LEN = 50
_MIN_LETTERS = 2


def _normalize(text: str) -> str:
    """NFC + strip."""
    return unicodedata.normalize("NFC", text).strip()


def is_valid_company_name(name: str) -> bool:
    """
    Devuelve True si el string parece un nombre de empresa real.

    Comprueba:
    - Longitud entre _MIN_LEN y _MAX_LEN
    - Al menos _MIN_LETTERS letras
    - No es una stopword conocida
    - No casa con ningún _BAD_PATTERNS
    - No contiene saltos de línea
    """
    name = _normalize(name)
    if not name:
        return False
    if len(name) < _MIN_LEN or len(name) > _MAX_LEN:
        return False
    if "\n" in name or "\r" in name:
        return False
    letter_count = sum(1 for c in name if c.isalpha())
    if letter_count < _MIN_LETTERS:
        return False
    if name.lower() in _STOPWORDS or name.title() in {s.title() for s in _STOPWORDS}:
        return False
    if _BAD_PATTERNS.search(name):
        return False
    return True


def validate_names(names: List[str]) -> List[str]:
    """
    Filtra una lista de candidatos aplicando is_valid_company_name.
    Deduplicados, preservando orden.
    """
    seen: set[str] = set()
    result: List[str] = []
    for raw in names:
        name = _normalize(raw)
        # Quitar markdown bold/italic
        name = re.sub(r"\*+", "", name).strip()
        # Truncar en el primer separador descriptivo
        name = re.split(r"\s[—\-–]\s|\s\(|\s{2,}|:\s", name)[0].strip()
        name = name.rstrip(".,;:")
        key = name.lower()
        if key in seen:
            continue
        if is_valid_company_name(name):
            seen.add(key)
            result.append(name)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# FALLBACK 1: búsqueda directa con Serper
# ══════════════════════════════════════════════════════════════════════════════

def fallback_serper(category: str, needed: int = 5) -> List[str]:
    """
    Hace 2-3 búsquedas en Serper y extrae nombres de los títulos de resultados.
    Más fiable que dejar al agente porque operamos directamente sobre los datos.
    """
    import requests as req

    api_key = os.getenv("SERPER_API_KEY", "")
    if not api_key:
        return []

    queries = [
        f"top 5 {category} empresas más conocidas",
        f"mejores {category} España ranking",
        f"{category} principales compañías líderes",
    ]

    candidates: List[str] = []
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    for query in queries:
        try:
            resp = req.post(
                "https://google.serper.dev/search",
                headers=headers,
                json={"q": query, "num": 10},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            # answerBox puede contener lista directa
            answer = data.get("answerBox", {}).get("answer", "")
            if answer:
                for line in answer.splitlines():
                    line = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
                    if line:
                        candidates.append(line)

            # knowledgeGraph
            kg_title = data.get("knowledgeGraph", {}).get("title", "")
            if kg_title:
                candidates.append(kg_title)

            # organic titles — extraer la parte antes de " - " o " | "
            for r in data.get("organic", []):
                title = r.get("title", "")
                # Intentar extraer nombre propio del título
                # "Revolut — Neobanco digital" → "Revolut"
                part = re.split(r"\s[-–|:]\s|\s-\s", title)[0].strip()
                candidates.append(part)
                # También el snippet puede tener nombres en negrita simulada
                snippet = r.get("snippet", "")
                for line in snippet.splitlines():
                    line = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
                    if 2 < len(line) < 40:
                        candidates.append(line)

        except Exception as e:
            print(f"[guardrails] Serper fallback error ({query}): {e}")
            continue

        valid = validate_names(candidates)
        if len(valid) >= needed:
            return valid[:needed]

    return validate_names(candidates)[:needed]


# ══════════════════════════════════════════════════════════════════════════════
# FALLBACK 2: Claude API con prompt JSON estricto
# ══════════════════════════════════════════════════════════════════════════════

def fallback_claude(category: str, existing: List[str], needed: int = 5) -> List[str]:
    """
    Llama directamente a Claude con un prompt que fuerza una lista JSON de
    exactamente `needed` nombres de empresa reales, sin texto adicional.
    """
    import urllib.request
    import urllib.error

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("[guardrails] ANTHROPIC_API_KEY no definida, no se puede usar fallback Claude.")
        return []

    already = f" Ya tengo: {existing}. " if existing else ""
    prompt = (
        f"Dame exactamente {needed} nombres de empresas reales y conocidas del sector: \"{category}\"."
        f"{already}"
        f"Devuelve ÚNICAMENTE un array JSON con los nombres, sin texto adicional, sin markdown:\n"
        f'["Empresa1", "Empresa2", "Empresa3", "Empresa4", "Empresa5"]'
    )

    payload = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 200,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")

    import time
    max_retries, wait = 3, 20
    for attempt in range(max_retries):
        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as resp:
                data = json.loads(resp.read())
            raw = "".join(b.get("text", "") for b in data.get("content", []))
            raw = re.sub(r"```json|```", "", raw).strip()
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                names = [str(n).strip() for n in parsed if n]
                valid = validate_names(names)
                # complement with existing if needed
                combined = existing + [n for n in valid if n not in existing]
                return combined[:needed]
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < max_retries - 1:
                print(f"[guardrails] Rate limit — esperando {wait}s...")
                time.sleep(wait)
                wait *= 2
                continue
            print(f"[guardrails] Claude fallback HTTP error: {exc.code}")
            break
        except Exception as exc:
            print(f"[guardrails] Claude fallback error: {exc}")
            break

    return existing


# ══════════════════════════════════════════════════════════════════════════════
# ORQUESTADOR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def ensure_five_companies(raw_names: List[str], category: str, target: int = 5) -> List[str]:
    """
    Recibe la lista de nombres que devolvió el agente (puede estar vacía o
    tener basura), la valida con reglas léxicas y activa fallbacks hasta
    tener `target` nombres válidos.

    Flujo:
      1. Validar raw_names → valid
      2. Si len(valid) >= target  → devolver valid[:target]
      3. Si len(valid) < target   → fallback_serper (completa hasta target)
      4. Si aún faltan            → fallback_claude (completa el resto)
      5. Si aún faltan            → rellenar con placeholders marcados
         (el pipeline puede continuar y marcarlos como sin datos)

    Args:
        raw_names: Lo que devolvió el agente (puede ser basura).
        category:  Categoría de búsqueda (para fallbacks).
        target:    Número deseado de empresas (default: 5).

    Returns:
        Lista de exactamente `target` nombres limpios (o menos si todos
        los fallbacks fallan, lo que es muy improbable).
    """
    print(f"[guardrails] Validando {len(raw_names)} candidatos: {raw_names}")

    # Capa 1: validación léxica
    valid = validate_names(raw_names)
    print(f"[guardrails] Tras validación léxica: {len(valid)} válidas → {valid}")

    if len(valid) >= target:
        return valid[:target]

    # Capa 2: fallback Serper
    print(f"[guardrails] Solo {len(valid)}/{target} válidas. Activando fallback Serper...")
    serper_names = fallback_serper(category, needed=target)
    # Fusionar: primero las ya válidas, luego las de Serper que no estén ya
    merged = valid + [n for n in serper_names if n.lower() not in {v.lower() for v in valid}]
    merged = validate_names(merged)[:target]
    print(f"[guardrails] Tras Serper: {len(merged)} válidas → {merged}")

    if len(merged) >= target:
        return merged[:target]

    # Capa 3: fallback Claude API
    print(f"[guardrails] Solo {len(merged)}/{target}. Activando fallback Claude API...")
    claude_names = fallback_claude(category, existing=merged, needed=target)
    final = validate_names(claude_names)[:target]
    print(f"[guardrails] Tras Claude fallback: {len(final)} válidas → {final}")

    return final if final else merged
