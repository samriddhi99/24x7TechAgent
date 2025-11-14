#!/usr/bin/env python3
"""
Call-Center Audio → RAG → Audio (+ chatbot .txt)
------------------------------------------------
- MP3 → speech-to-text
- Query SearchEngine.process_query()
- Clean for call center voice (concise, safe)
- TTS to MP3
- Save a clean .txt for chatbot (no call-center framing)

Usage from Flask (example below):
    from audio_processing_service import callcenter_answer_from_mp3
"""

import os
import re
import io
import tempfile
import shutil
from typing import List, Tuple, Dict, Any, Optional

import speech_recognition as sr
from pydub import AudioSegment
from gtts import gTTS

# Import RAG pipeline function from your app
from .search_engine_webApp import process_query

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
    patterns = [
        r"^\s*okay, here[’']?s .*?$",
        r"^\s*okay[, ]+let['’]s.*?$",
        r"^\s*based on (the )?provided documentation.*?$",
        r"^\s*llm[- ]?enhanced insights.*?$",
        r"^\s*disclaimer\s*:.*?$",
        r"^\s*resources for learning\s*:.*?$",
        r"^\s*sources used\s*:.*?$",
    ]
    lines = [ln for ln in text.splitlines()]
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
    return "\n".join([c for c in cleaned if c.strip()]).strip()

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
        # Treat “Section: ” headings as paragraph breaks (not items)
        is_heading = bool(re.match(r"^\s*[^:]{2,}:\s*$", line))
        if m_b or m_n:
            items.append((m_b.group(1) if m_b else m_n.group(1)).strip())
        else:
            flush()
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
        try:
            return f"{int(m.group(0)):,}"
        except Exception:
            return m.group(0)
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

def call_center_style(answer_text: str, lang: str = "EN", pronunciation_map: Optional[Dict[str, str]] = None) -> str:
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
    txt = re.sub(r"^\s*important note\s*:.*?$", "", txt, flags=re.I | re.M).strip()
    txt = dedupe_and_limit(txt, max_sentences=5)  # Shorter for call flows

    combined = " ".join([opening, txt, handoff, closing])
    combined = re.sub(r"\s{2,}", " ", combined).strip()

    # Optional pronunciations (e.g., "CC-IN2P3": "C C I N two P three")
    if pronunciation_map:
        for k, v in pronunciation_map.items():
            combined = combined.replace(k, v)

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
    wav_path = None
    try:
        wav_path = tempfile.mktemp(prefix="ccrag_", suffix=".wav")
        AudioSegment.from_mp3(mp3_path).export(wav_path, format="wav")
        r = sr.Recognizer()
        with sr.AudioFile(wav_path) as src:
            audio = r.record(src)
        return r.recognize_google(audio)
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass

def mp3_bytes_to_text(audio_bytes: bytes) -> str:
    """Transcribe MP3 given raw bytes (useful for Flask uploads)."""
    ensure_ffmpeg_available()
    tmp_mp3 = tempfile.mktemp(prefix="ccrag_", suffix=".mp3")
    with open(tmp_mp3, "wb") as f:
        f.write(audio_bytes)
    try:
        return mp3_to_text(tmp_mp3)
    finally:
        try:
            os.remove(tmp_mp3)
        except Exception:
            pass

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

# ----------------------------- Public API -----------------------------

def callcenter_answer_from_mp3(
    input_mp3_path: str,
    output_mp3_path: str,
    *,
    lang: str = "EN",
    mode: str = "hybrid",
    model: str = "llama3.1:8b",
    save_text_path: Optional[str] = None,
    pronunciation_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Main callable for Flask/servers: MP3 path → transcript → RAG → clean → MP3 (and .txt).
    Returns a dict with transcript, answer_text, speak_text, sources, latency, and file paths.
    """
    if not os.path.exists(input_mp3_path):
        raise FileNotFoundError(f"Input file not found: {input_mp3_path}")

    transcript = mp3_to_text(input_mp3_path)

    answer_text, sources, latency = run_rag_pipeline(transcript, model=model, lang=lang, mode=mode)

    speak_text = call_center_style(answer_text, lang=lang, pronunciation_map=pronunciation_map)
    chat_txt = chatbot_text(answer_text)

    # Save chatbot text if requested
    if save_text_path:
        os.makedirs(os.path.dirname(save_text_path) or ".", exist_ok=True)
        with open(save_text_path, "w", encoding="utf-8") as f:
            f.write(chat_txt + "\n")

    # TTS
    tts_lang = "fr" if lang.upper().startswith("FR") else "en"
    os.makedirs(os.path.dirname(output_mp3_path) or ".", exist_ok=True)
    text_to_mp3(speak_text, output_mp3_path, tts_lang=tts_lang)

    return {
        "transcript": transcript,
        "answer_text": answer_text,
        "speak_text": speak_text,
        "sources": sources,            # list of {url, text, score}
        "latency": latency,            # seconds
        "output_mp3_path": output_mp3_path,
        "chat_text_path": save_text_path,
    }

def callcenter_answer_from_bytes(
    path,
    output_dir: str,
    filename_base: str = "rag_output",
    *,
    lang: str = "EN",
    mode: str = "hybrid",
    model: str = "llama3.1:8b",
    save_text: bool = True,
    pronunciation_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Same as above, but accepts raw MP3 bytes (e.g., Flask file upload).
    Saves files into output_dir with base name.
    """
    
    with open(path, "rb") as infile:
        audio_bytes = infile.read()

    os.makedirs(output_dir, exist_ok=True)
    input_mp3_path = os.path.join(output_dir, f"{filename_base}.mp3")

    with open(input_mp3_path, "wb") as f:
        f.write(audio_bytes)
    output_mp3_path = os.path.join(output_dir, f"{filename_base}_answer.mp3")
    save_text_path = os.path.join(output_dir, f"{filename_base}.txt") if save_text else None

    return callcenter_answer_from_mp3(
        input_mp3_path=input_mp3_path,
        output_mp3_path=output_mp3_path,
        lang=lang,
        mode=mode,
        model=model,
        save_text_path=save_text_path,
        pronunciation_map=pronunciation_map,
    )

# ----------------------------- Minimal Flask example -----------------------------
# (You can remove this section if you already have your Flask app elsewhere.)

"""
from flask import Flask, request, send_file, jsonify
import tempfile

app = Flask(__name__)

@app.post("/ask-audio")
def ask_audio():
    # Expect multipart/form-data with field name "audio"
    if "audio" not in request.files:
        return jsonify({"error": "Missing 'audio' file"}), 400

    f = request.files["audio"]
    lang = request.form.get("lang", "EN")
    mode = request.form.get("mode", "hybrid")
    model = request.form.get("model", "llama3.1:8b")

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_bytes = f.read()
        result = callcenter_answer_from_bytes(
            audio_bytes,
            output_dir=tmpdir,
            filename_base="response",
            lang=lang,
            mode=mode,
            model=model,
            pronunciation_map={"CC-IN2P3": "C C I N two P three"},
        )

        # Example JSON response (paths are temp; stream the MP3 back)
        return jsonify({
            "transcript": result["transcript"],
            "answer_text": result["answer_text"],
            "speak_text": result["speak_text"],
            "latency": result["latency"],
            "sources": result["sources"],
        })

# If you'd rather return the MP3 file directly:
# return send_file(result["output_mp3_path"], mimetype="audio/mpeg", as_attachment=True, download_name="answer.mp3")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5060, debug=True)
"""
