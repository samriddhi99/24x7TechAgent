#!/usr/bin/env python3
"""
Call-Center Audio → RAG → Audio (+ chatbot .txt) WITH AUTO-ESCALATION
----------------------------------------------------------------------
- MP3 → speech-to-text
- Query SearchEngine.process_query() with confidence scoring
- If confidence low → escalate to human agent
- If confidence good → clean for call center voice (concise, safe)
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
from typing import List, Tuple, Dict

import speech_recognition as sr
from pydub import AudioSegment
from gtts import gTTS

# Import RAG pipeline function from your app (now returns confidence info)
from search_engine_webApp import process_query

# ----------------------------- Utilities -----------------------------

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9«""])')

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
    # Kill common "meta" lead-ins and sections
    patterns = [
        r"^\s*okay, here['']?s .*?$",
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
        # end skip when we hit an empty line or a new "section-ish" line
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
        # Also treat "Section: " headings as paragraph breaks (not items)
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

# ----------------------------- Call Center Templates -----------------------------

TEMPLATES = {
    "EN": {
        "opening": "Thanks for calling. Here's a concise answer.",
        "closing": "Is there anything else I can help you with?",
        "handoff": "If you need more detail, I can send a summary by email or connect you to a specialist.",
        # NEW: Escalation message for transferring to human
        "escalation": "I understand your question. However, I'd like to connect you with one of our specialists who can provide more detailed assistance. Please hold while I transfer your call. Thank you for your patience.",
    },
    "FR": {
        "opening": "Merci de votre appel. Voici une réponse concise.",
        "closing": "Puis-je vous aider pour autre chose ?",
        "handoff": "Si besoin de plus de détails, je peux envoyer un résumé par e-mail ou vous transférer à un spécialiste.",
        # NEW: Escalation message for transferring to human
        "escalation": "Je comprends votre question. Cependant, j'aimerais vous mettre en relation avec l'un de nos spécialistes qui pourra vous fournir une assistance plus détaillée. Veuillez patienter pendant que je transfère votre appel. Merci de votre patience.",
    },
}

def call_center_style(answer_text: str, lang: str = "EN") -> str:
    """Format answer for call center with opening/closing"""
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
    # remove "important note" lines
    txt = re.sub(r"^\s*important note\s*:.*?$", "", txt, flags=re.I | re.M).strip()
    txt = dedupe_and_limit(txt, max_sentences=7)  # ~20–30s

    combined = " ".join([opening, txt, handoff, closing])
    combined = re.sub(r"\s{2,}", " ", combined).strip()
    return combined

def escalation_message(lang: str = "EN") -> str:
    """Get the escalation message for transferring to human agent"""
    lang = "FR" if str(lang).upper().startswith("FR") else "EN"
    return TEMPLATES[lang]["escalation"]

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

def run_rag_pipeline(
    query_text: str, 
    model: str, 
    lang: str, 
    mode: str
) -> Tuple[str, list, float, Dict]:
    """
    Run RAG pipeline with confidence scoring
    
    Returns:
        answer: str - The AI's answer
        sources: list - Source documents used
        latency: float - Processing time
        confidence_info: dict - Confidence metrics including 'should_escalate'
    """
    answer, sources, latency, confidence_info = process_query(
        user_question=query_text,
        llm_model=model,
        lang=lang,
        mode=mode
    )
    return answer, sources, latency, confidence_info

def text_to_mp3(text: str, output_path: str, tts_lang: str = "en") -> None:
    tts = gTTS(text=text, lang=tts_lang)
    tts.save(output_path)

# ----------------------------- Orchestrator -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Call-center MP3 → RAG → MP3 (+chatbot .txt) with auto-escalation")
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

    print("=" * 60)
    print("🤖 CALL CENTER AI ASSISTANT WITH AUTO-ESCALATION")
    print("=" * 60 + "\n")

    print(f"🎧 Transcribing: {args.input}")
    #transcript = mp3_to_text(args.input)
    transcript = "What is shopify and how does it work?"  # Uncomment for testing without audio
    print(f"📝 Transcript: {transcript}\n")

    print(f"🤖 RAG ({args.mode}, {args.model}) …")
    answer_text, sources, latency, confidence_info = run_rag_pipeline(
        transcript, 
        model=args.model, 
        lang=args.lang, 
        mode=args.mode
    )
    
    print(f"✅ RAG answer in {latency:.2f}s")
    print(f"📊 Confidence: {confidence_info['confidence_score']:.0%}")
    print(f"📋 Reason: {confidence_info['reason']}\n")

    # ===== ESCALATION LOGIC =====
    if confidence_info['should_escalate']:
        print("🔄 LOW CONFIDENCE - Escalating to human agent")
        print("=" * 60)
        
        # Use escalation message instead of AI answer
        speak_text = escalation_message(lang=args.lang)
        chat_txt = f"[ESCALATED TO HUMAN AGENT]\n\nReason: {confidence_info['reason']}\nConfidence: {confidence_info['confidence_score']:.0%}\n\nOriginal Question: {transcript}"
        
        print("🔊 Escalation message:")
        print(f"   {speak_text}\n")
        
    else:
        print("✅ HIGH CONFIDENCE - Providing AI answer")
        print("=" * 60)
        
        # Format answer for call center
        print("🧹 Formatting for call-center voice …")
        speak_text = call_center_style(answer_text, lang=args.lang)
        chat_txt = chatbot_text(answer_text)
        
        print("🔊 Final talk track:")
        print(f"   {speak_text[:100]}...\n")

    # Save chatbot-friendly text with proper metadata
    with open(base_txt, "w", encoding="utf-8") as f:
        f.write(chat_txt + "\n")
        f.write(f"\n--- Metadata ---\n")
        # FIX: Use the actual float value, not formatted percentage
        f.write(f"Confidence: {confidence_info['confidence_score'] * 100:.1f}%\n")
        f.write(f"Should Escalate: {confidence_info['should_escalate']}\n")
        f.write(f"Reason: {confidence_info['reason']}\n")
        f.write(f"Escalated: {'Yes' if confidence_info['should_escalate'] else 'No'}\n")
        if sources:
            f.write(f"\nSources:\n")
            for src in sources:
                f.write(f"- {src['url']} (score: {src['score']})\n")
    print(f"📝 Saved chatbot text: {base_txt}")

    # Generate TTS
    tts_lang = "fr" if args.lang.upper().startswith("FR") else "en"
    text_to_mp3(speak_text, args.output, tts_lang=tts_lang)
    print(f"💾 Saved audio: {args.output}")

    print("\n" + "=" * 60)
    print("✅ PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Action: {'🔄 ESCALATE TO HUMAN' if confidence_info['should_escalate'] else '✅ AI ANSWER PROVIDED'}")
    print(f"Confidence: {confidence_info['confidence_score']:.0%}")
    print(f"Output Audio: {args.output}")
    print(f"Output Text: {base_txt}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()