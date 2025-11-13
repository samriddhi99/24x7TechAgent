#!/usr/bin/env python3
"""
Call-Center Audio → RAG → Audio (+ chatbot .txt)
------------------------------------------------
- MP3 → speech-to-text
- Query SearchEngine.process_query()
- Clean for call center voice (concise, safe)
- TTS to MP3
- Save a clean .txt for chatbot (no call-center framing)

Usage:
  python audio_processingv2.1.py --input input.mp3 --output rag_output.mp3 \
    --lang EN --mode hybrid --model "gemma3:4b" --save-text rag_output.txt
"""

import os
import re
import argparse
import shutil
from typing import List, Tuple

import speech_recognition as sr
from pydub import AudioSegment
from gtts import gTTS

# Import RAG pipeline function from your app
from search_engine_webApp import process_query

# ----------------------------- Utilities -----------------------------

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9«“"])')

def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if s and s.strip()]

def join_sentences(sents: List[str]) -> str:
    out = []
    for s in sents:
        s = re.sub(r'\s+', ' ', s).strip()
        if not s.endswith(('.', '!', '?')):
            s += '.'
        out.append(s)
    return ' '.join(out)

def strip_markdown_and_links(text: str) -> str:
    # code blocks + inline code
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # bold/italic
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    # headings/quotes
    text = re.sub(r"^\s{0,3}#{1,6}\s*(.+)$", r"\1: ", text, flags=re.M)
    text = re.sub(r"^\s*>\s*", "", text, flags=re.M)
    # [label](url) -> label, raw URLs -> drop
    text = re.sub(r"\[([^\]]+)\]\((http[s]?://[^\)]+)\)", r"\1", text)
    text = re.sub(r"(http[s]?://\S+)", "", text)
    return text

# Remove boilerplate sections commonly returned by RAG/LLM answers
def remove_meta_sections(text: str) -> str:
    # Kill common “meta” lead-ins and sections
    patterns = [
        r"^\s*okay, here[’']?s .*?$",
        r"^\s*llm[- ]?enhanced insights.*?$",
        r"^\s*disclaimer\s*:.*?$",
        r"^\s*resources for learning\s*:.*?$",
        r"^\s*sources used\s*:.*?$",
    ]
    lines = [ln for ln in text.splitlines() if ln.strip()]
    cleaned = []
    skip_block = False
    for ln in lines:
        if any(re.match(p, ln.strip(), flags=re.I) for p in patterns):
            skip_block = True
            continue
        # end skip when we hit an empty line or a new “section-ish” line
        if skip_block and (not ln.strip() or re.match(r"^\s*[A-Z].*?:\s*$", ln)):
            skip_block = False
            continue
        if not skip_block:
            cleaned.append(ln)
    return "\n".join(cleaned).strip()

def flatten_lists(text: str) -> str:
    """Turn bullets/numbers into one ordered spoken list per block."""
    lines = (text or "").splitlines()
    items, out = [], []

    def flush():
        nonlocal items, out
        if not items:
            return
        ords = ["First", "Next", "Then", "Also", "Finally"]
        parts = []
        for i, it in enumerate(items):
            label = ords[i] if i < len(ords) else "Also"
            it = it.strip().rstrip('.')
            parts.append(f"{label}, {it}.")
        out.append(" ".join(parts))
        items = []

    for line in lines:
        m_b = re.match(r"^\s*[-*•]\s+(.*)$", line)
        m_n = re.match(r"^\s*\d+[\.\)]\s+(.*)$", line)
        # Also treat “Section: ” headings as paragraph breaks (not items)
        is_heading = bool(re.match(r"^\s*[^:]{2,}:\s*$", line))
        if m_b or m_n:
            items.append((m_b.group(1) if m_b else m_n.group(1)).strip())
        else:
            flush()
            # Drop bare headings; content speaks itself
            if not is_heading:
                out.append(line)
    flush()
    return "\n".join(out)

# PII redaction
_PII_PATTERNS = [
    (re.compile(r"\b(?:\+?\d[\s\-\.()]*){7,}\b"), "[redacted phone]"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[redacted email]"),
    (re.compile(r"\b(?:\d[ -]*?){13,19}\b"), "[redacted account]"),
]
def redact_pii(text: str) -> str:
    for pat, repl in _PII_PATTERNS:
        text = pat.sub(repl, text)
    return text

# Tone
_PROFANITY = re.compile(r"\b(fuck|shit|damn|bitch|asshole)\b", re.I)
def clean_tone(text: str) -> str:
    return _PROFANITY.sub("[expletive]", text)

def speakable_numbers(text: str) -> str:
    def _fmt(m):
        try: return f"{int(m.group(0)):,}"
        except: return m.group(0)
    text = re.sub(r"\b\d{5,}\b", _fmt, text)
    text = text.replace("%", " percent")
    return text

def dedupe_and_limit(text: str, max_sentences: int = 7) -> str:
    seen, kept = set(), []
    for s in split_sentences(text):
        key = re.sub(r"\s+", " ", s).strip().lower()
        if key and key not in seen:
            kept.append(s)
            seen.add(key)
        if len(kept) >= max_sentences:
            break
    return join_sentences(kept)

TEMPLATES = {
    "EN": {
        "opening": "Thanks for calling. Here’s a concise answer.",
        "closing": "Is there anything else I can help you with?",
        "handoff": "If you need more detail, I can send a summary by email or connect you to a specialist.",
    },
    "FR": {
        "opening": "Merci de votre appel. Voici une réponse concise.",
        "closing": "Puis-je vous aider pour autre chose ?",
        "handoff": "Si besoin de plus de détails, je peux envoyer un résumé par e-mail ou vous transférer à un spécialiste.",
    },
}

def call_center_style(answer_text: str, lang: str = "EN") -> str:
    lang = "FR" if str(lang).upper().startswith("FR") else "EN"
    opening = TEMPLATES[lang]["opening"]
    closing = TEMPLATES[lang]["closing"]
    handoff = TEMPLATES[lang]["handoff"]

    txt = answer_text or ""
    txt = strip_markdown_and_links(txt)
    txt = remove_meta_sections(txt)
    txt = flatten_lists(txt)
    txt = redact_pii(txt)
    txt = clean_tone(txt)
    txt = speakable_numbers(txt)
    # remove “important note” lines
    txt = re.sub(r"^\s*important note\s*:.*?$", "", txt, flags=re.I | re.M).strip()
    txt = dedupe_and_limit(txt, max_sentences=7)  # ~20–30s

    combined = " ".join([opening, txt, handoff, closing])
    combined = re.sub(r"\s{2,}", " ", combined).strip()
    return combined

def chatbot_text(answer_text: str) -> str:
    """
    Cleaned content for chat surfaces: NO call-center framing,
    no URLs/markdown, concise and neutral tone.
    """
    txt = answer_text or ""
    txt = strip_markdown_and_links(txt)
    txt = remove_meta_sections(txt)
    txt = flatten_lists(txt)
    txt = redact_pii(txt)
    txt = clean_tone(txt)
    txt = speakable_numbers(txt)
    txt = re.sub(r"^\s*important note\s*:.*?$", "", txt, flags=re.I | re.M).strip()
    txt = dedupe_and_limit(txt, max_sentences=8)
    return txt

# ----------------------------- Audio I/O -----------------------------

def ensure_ffmpeg_available() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise RuntimeError(
            "FFmpeg not found: missing {}. Install it and ensure it's on PATH.\n"
            "macOS: brew install ffmpeg".format(", ".join(missing))
        )

def mp3_to_text(mp3_path: str) -> str:
    ensure_ffmpeg_available()
    wav_path = "._temp_callcenter.wav"
    AudioSegment.from_mp3(mp3_path).export(wav_path, format="wav")
    r = sr.Recognizer()
    with sr.AudioFile(wav_path) as src:
        audio = r.record(src)
    try:
        return r.recognize_google(audio)
    finally:
        try: os.remove(wav_path)
        except Exception: pass

def run_rag_pipeline(query_text: str, model: str, lang: str, mode: str) -> Tuple[str, list, float]:
    answer, sources, latency = process_query(
        user_question=query_text,
        llm_model=model,
        lang=lang,
        mode=mode
    )
    return answer, sources, latency

def text_to_mp3(text: str, output_path: str, tts_lang: str = "en") -> None:
    tts = gTTS(text=text, lang=tts_lang)
    tts.save(output_path)

# ----------------------------- Orchestrator -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Call-center MP3 → RAG → MP3 (+chatbot .txt)")
    parser.add_argument("--input", "-i", required=True, help="Input MP3 file (caller question)")
    parser.add_argument("--output", "-o", default="rag_output.mp3", help="Output MP3 file (spoken answer)")
    parser.add_argument("--save-text", default=None, help="Path to save chatbot text (default: output basename .txt)")
    parser.add_argument("--lang", default="EN", choices=["EN", "FR"], help="RAG language")
    parser.add_argument("--mode", default="hybrid", choices=["rag_only", "hybrid", "llm_only"], help="Query mode")
    parser.add_argument("--model", default="gemma3:4b", help="Ollama model name")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    base_txt = args.save_text or os.path.splitext(args.output)[0] + ".txt"

    print(f"🎧 Transcribing: {args.input}")
    # transcript = mp3_to_text(args.input)
    transcript = "What are Python’s main data types?"  # Placeholder for testing
    print("📝 Transcript:\n", transcript, "\n")

    print(f"🤖 RAG ({args.mode}, {args.model}) …")
    answer_text, sources, latency = run_rag_pipeline(transcript, model=args.model, lang=args.lang, mode=args.mode)
    print(f"✅ RAG answer in {latency:.2f}s\n{answer_text}\n")

    # Build voice-friendly text
    print("🧹 Formatting for call-center voice …")
    speak_text = call_center_style(answer_text, lang=args.lang)
    print("🔊 Final talk track:\n", speak_text, "\n")

    # Save chatbot-friendly text (no call-center framing)
    chat_txt = chatbot_text(answer_text)
    with open(base_txt, "w", encoding="utf-8") as f:
        f.write(chat_txt + "\n")
    print(f"📝 Saved chatbot text: {base_txt}")

    # TTS
    tts_lang = "fr" if args.lang.upper().startswith("FR") else "en"
    text_to_mp3(speak_text, args.output, tts_lang=tts_lang)
    print(f"💾 Saved audio: {args.output}")

if __name__ == "__main__":
    main()
