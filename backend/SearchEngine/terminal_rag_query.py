import json
import numpy as np
import requests
import tempfile
import os
import webbrowser
import time
from markdown2 import markdown
from xhtml2pdf import pisa
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from functools import lru_cache
from rich.console import Console
from rich.markdown import Markdown

console = Console()

TOP_K = 50
TOP_K_RELEVANT = 5
THRESHOLD_GOOD = 0.70
DEFAULT_LLM_MODEL = "gemma3:4b"
DEFAULT_LANGUAGE = "EN"
DEFAULT_QUERY_MODE = "rag_only"

client = PersistentClient(path="./chroma_db")
collection = client.get_collection(name="web_chunks")

# --- Embedding with cache ---
embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

@lru_cache(maxsize=500)
def cached_embed(text):
    return embed_model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]

def embed_texts(texts):
    return [cached_embed(text) for text in texts]

def call_ollama_llm(prompt, model, temperature=0.1):
    try:
        payload = {"model": model, "prompt": prompt, "options": {"temperature": temperature}}
        response = requests.post("http://localhost:11434/api/generate", json=payload, stream=True)
        if response.status_code == 200:
            return ''.join(json.loads(line.decode("utf-8"))['response']
                           for line in response.iter_lines() if line and 'response' in json.loads(line.decode("utf-8")))
        else:
            return f"LLM API error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"LLM API exception: {str(e)}"

def get_translations(lang):
    return {
        "intro": "Vous êtes un assistant expert qui aide les utilisateurs à comprendre de la documentation technique." if lang == "FR"
                 else "You are an expert assistant helping users understand technical documentation.",
        "instr_rag": "Fournissez une réponse détaillée, structurée et pratique uniquement à partir de la documentation fournie." if lang == "FR"
                     else "Provide a detailed, structured, and practical answer using only the provided documentation.",
        "instr_hybrid": "Ajoutez des compléments issus du LLM si pertinent, en les identifiant clairement." if lang == "FR"
                        else "If relevant, enhance the response with complementary LLM knowledge and clearly indicate what part comes from the LLM.",
        "no_docs": "⚠️ Aucun document pertinent trouvé." if lang == "FR"
                   else "⚠️ No relevant documents found."
    }

def process_query(user_question, llm_model, lang, mode=DEFAULT_QUERY_MODE):
    start_time = time.time()
    temperature = 0.1 if mode == "rag_only" else 0.4 if mode == "hybrid" else 0.7
    t = get_translations(lang)

    if mode == "llm_only":
        prompt = f"{t['intro']}\n\nQuestion: {user_question}\n\nAnswer strictly using the LLM's internal knowledge."
        duration = time.time() - start_time
        return call_ollama_llm(prompt, llm_model, temperature), [], duration

    query_emb = embed_texts([user_question])[0]
    results = collection.query(query_embeddings=[query_emb], n_results=TOP_K, include=["documents", "metadatas", "distances"])

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    scores = results.get("distances", [[]])[0]

    relevant = [(doc, meta, score) for doc, meta, score in zip(docs, metas, scores) if score >= THRESHOLD_GOOD]

    if not relevant:
        duration = time.time() - start_time
        return t["no_docs"], [], duration

    # ⏱️ Sélection des n meilleurs documents seulement
    relevant = sorted(relevant, key=lambda x: x[2], reverse=True)[:TOP_K_RELEVANT]

    page_map = {}
    for doc, meta, score in relevant:
        url = meta.get("url", "")
        page_map.setdefault(url, {"text": "", "score": round(score, 4)})
        page_map[url]["text"] += "\n" + doc

    page_contexts = [{"url": url, "text": d["text"], "score": d["score"]} for url, d in page_map.items()]
    all_text = "\n\n".join(p["text"] for p in page_contexts)
    all_urls = [p["url"] for p in page_contexts]

    prompt = f"""{t['intro']}

Sources:
{chr(10).join('- ' + url for url in all_urls)}

Documentation:
{all_text}

Question: {user_question}

{t['instr_rag']}"""
    if mode == "hybrid":
        prompt += f"\n{t['instr_hybrid']}"

    duration = time.time() - start_time
    return call_ollama_llm(prompt, llm_model, temperature), page_contexts, duration

def generate_pdf(content, filename="rag_answer.pdf"):
    html_template = f"""
    <html>
    <head>
        <meta charset='UTF-8'>
        <style>
            body {{ font-family: Helvetica, sans-serif; line-height: 1.4; font-size: 12pt; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            code {{ background-color: #f4f4f4; padding: 2px 4px; }}
        </style>
    </head>
    <body>{markdown(content)}</body>
    </html>
    """
    output_path = os.path.join(os.getcwd(), filename)
    with open(output_path, "wb") as f:
        pisa.CreatePDF(html_template, dest=f)
    return output_path

def main():
    print("📚 Welcome to RAG Terminal Assistant")
    llm_model = DEFAULT_LLM_MODEL
    mode = DEFAULT_QUERY_MODE
    lang = DEFAULT_LANGUAGE

    while True:
        print("\n" + "-" * 40)
        question = input("🧑 Your question (or 'exit'): ").strip()
        if question.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        print(f"\n⏳ Mode: {mode.upper()} | Langue: {lang} — Please wait...")
        answer, sources, latency = process_query(question, llm_model, lang, mode)

        print("\n🤖 Assistant (rendered Markdown):\n")
        console.print(Markdown(answer))
        print(f"\n⏱️ Answered in {latency:.2f} seconds")

        if sources:
            print("\n🔗 Sources:")
            for s in sorted(sources, key=lambda x: x["score"], reverse=True):
                print(f"- {s['url']} (score: {s['score']})")

        if input("\n💾 Export to PDF? (y/n): ").strip().lower() == "y":
            content = f"# Question\n{question}\n\n# Answer\n{answer}\n\n# Sources\n" + \
                      "\n".join([f"- {s['url']} (score: {s['score']})" for s in sources]) + \
                      f"\n\n⏱️ Answered in {latency:.2f} seconds"
            path = generate_pdf(content)
            print(f"✅ PDF saved: {path}")
            try:
                webbrowser.open(f"file://{path}")
            except:
                print("⚠️ Could not open the PDF automatically.")

        if input("\n⚙️ Change mode/lang? (y/n): ").strip().lower() == "y":
            mode_input = input("Mode (rag_only / hybrid / llm_only): ").strip().lower()
            if mode_input in ["rag_only", "hybrid", "llm_only"]:
                mode = mode_input
            lang_input = input("Language (EN / FR): ").strip().upper()
            if lang_input in ["EN", "FR"]:
                lang = lang_input

if __name__ == "__main__":
    main()