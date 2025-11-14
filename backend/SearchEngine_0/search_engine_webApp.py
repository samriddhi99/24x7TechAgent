# -----------------------------------------------------------------------------
# Enhanced search_engine_webApp.py with Simple Confidence-Based Escalation
# -----------------------------------------------------------------------------

import json
import numpy as np
from dash import Dash, html, dcc, Input, Output, State, ctx
import dash_bootstrap_components as dbc
from chromadb import PersistentClient
from xhtml2pdf import pisa
from markdown2 import markdown
import tempfile
import base64
import requests
import time
import re
from functools import lru_cache
from sentence_transformers import SentenceTransformer

# --- Configuration (in search_engine_webApp.py) ---
TOP_K = 30
THRESHOLD_GOOD = 0.70  # Keep this - it's a distance threshold
DEFAULT_LLM_MODEL = "gemma3:4b"
DEFAULT_LANGUAGE = "EN"
DEFAULT_QUERY_MODE = "rag_only"
MAX_CHARS = 8000
TOP_K_RELEVANT = 20

# ADJUSTED: Lower threshold for escalation (allow more answers through)
CONFIDENCE_THRESHOLD = 0.35  # Was 0.60, now 0.35
                              # Only escalate on truly bad retrievals

# ADJUSTED: More lenient retrieval threshold
MIN_RETRIEVAL_SCORE = 0.15  # Was 0.30, now 0.15
                            # Don't escalate unless retrieval is terrible


# --- Load Chroma Collection ---
client = PersistentClient(path="./chroma_db")
collection = client.get_collection(name="web_chunks")

# --- Embedding Utility (cached + in-memory) ---
embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

@lru_cache(maxsize=500)
def cached_embed(text):
    return embed_model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]

def embed_texts(texts):
    return [cached_embed(text) for text in texts]

# --- LLM Call ---
def call_ollama_llm(prompt, model, temperature=0.1):
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "options": {"temperature": temperature}
        }
        response = requests.post("http://localhost:11434/api/generate", json=payload, stream=True)
        if response.status_code == 200:
            answer_parts = []
            for line in response.iter_lines():
                if line:
                    try:
                        json_line = json.loads(line.decode("utf-8"))
                        if "response" in json_line:
                            answer_parts.append(json_line["response"])
                    except json.JSONDecodeError:
                        pass
            return ''.join(answer_parts)
        else:
            return f"LLM API error: {response.status_code}"
    except Exception as e:
        return f"LLM API exception: {str(e)}"

# --- Translations ---
def get_translations(lang):
    if lang == "FR":
        return {
            "ask_placeholder": "Posez votre question...",
            "model_label": "Modèle",
            "language_label": "Langue",
            "mode_label": "Mode",
            "submit": "Soumettre",
            "clear": "Effacer",
            "show_sources": "Afficher les sources",
            "you": "🧑 Vous",
            "assistant": "🤖 Assistant",
            "sources_used": "Sources utilisées :",
            "download_pdf": "📄 Télécharger PDF",
            "no_question": "❗ Veuillez entrer une question.",
            "no_relevant_docs": "⚠️ Aucun document pertinent trouvé dans notre base de connaissances.",
            "escalation_message": "🔄 Je ne suis pas suffisamment confiant pour répondre à cette question. Je vous transfère à un agent humain qui pourra mieux vous aider.",
            "low_confidence": "Confiance faible",
            "medium_confidence": "Confiance moyenne",
            "high_confidence": "Haute confiance"
        }
    else:
        return {
            "ask_placeholder": "Ask your question...",
            "model_label": "Model",
            "language_label": "Language",
            "mode_label": "Mode",
            "submit": "Submit",
            "clear": "Clear Output",
            "show_sources": "Show sources",
            "you": "🧑 You",
            "assistant": "🤖 Assistant",
            "sources_used": "Sources used:",
            "download_pdf": "📄 Download PDF",
            "no_question": "❗ Please enter a question.",
            "no_relevant_docs": "⚠️ No relevant documents found in our knowledge base.",
            "escalation_message": "🔄 I'm not confident enough to answer this question. Let me transfer you to a human agent who can better assist you.",
            "low_confidence": "Low confidence",
            "medium_confidence": "Medium confidence",
            "high_confidence": "High confidence"
        }

# ============================================================================
# SIMPLIFIED CONFIDENCE SCORING
# ============================================================================

def check_answer_confidence(answer: str, source_scores: list, lang: str) -> dict:
    """
    Adjusted confidence scoring with lower thresholds for better UX
    
    Key changes:
    - Lower overall confidence threshold (0.35 vs 0.60)
    - Lower retrieval threshold (0.15 vs 0.30)
    - More forgiving distance-to-similarity conversion
    - Only escalate on clear failure signals
    
    Returns:
        {
            'confidence_score': float (0-1),
            'should_escalate': bool,
            'reason': str
        }
    """
    
    # Uncertainty phrases (keep comprehensive)
    if lang == "FR":
        uncertainty_phrases = [
            "je ne sais pas",
            "je n'ai pas d'information",
            "je ne peux pas",
            "je ne suis pas sûr",
            "aucune information",
            "pas d'information",
            "je ne trouve pas",
            "pas dans la documentation",
            "je ne peux pas déterminer",
            "impossible de",
            "je n'ai pas trouvé",
        ]
    else:
        uncertainty_phrases = [
            "i don't know",
            "i'm not sure",
            "i don't have",
            "i cannot have",
            "no information",
            "not enough information",
            "can't find",
            "cannot find",
            "unable to",
            "not available",
            "not in the documentation",
            "cannot answer",
            "cannot determine",
            "can't determine",
            "i cannot",
            "cannot provide",
            "not contain",
            "does not contain",
            "doesn't contain",
            "insufficient information",
            "not possible to",
        ]
    
    answer_lower = answer.lower()
    
    # Check 1: Uncertainty detection (40% weight, reduced from 50%)
    has_uncertainty = any(phrase in answer_lower for phrase in uncertainty_phrases)
    uncertainty_score = 0.0 if has_uncertainty else 1.0
    
    # Check 2: Document retrieval quality (60% weight, increased from 50%)
    # IMPROVED: More forgiving distance-to-similarity conversion
    if source_scores and len(source_scores) > 0:
        # NEW: Use exponential decay for better score distribution
        # This gives more credit to distances around 10-15 (typical for your data)
        similarity_scores = [
            max(0.0, 1.0 - (score / 20.0))  # Linear decay, 0 at distance 20
            for score in source_scores
        ]
        
        # Average of top 5 documents
        top_similarities = sorted(similarity_scores, reverse=True)[:5]
        avg_similarity = sum(top_similarities) / len(top_similarities)
        retrieval_score = avg_similarity
        
        # Debug info
        print(f"DEBUG - Raw distances: {[f'{s:.2f}' for s in source_scores[:5]]}")
        print(f"DEBUG - Converted similarities: {[f'{s:.3f}' for s in similarity_scores[:5]]}")
        print(f"DEBUG - Avg similarity: {avg_similarity:.3f}")
    else:
        retrieval_score = 0.0
    
    # Calculate overall confidence
    # NEW: Weighted more toward retrieval quality (60/40 vs 50/50)
    confidence_score = (0.40 * uncertainty_score) + (0.60 * retrieval_score)
    confidence_score = max(0.0, min(1.0, confidence_score))
    
    # Determine if we should escalate
    # NEW: More lenient - only escalate on clear failures
    should_escalate = (
        (has_uncertainty and retrieval_score < MIN_RETRIEVAL_SCORE) or  # Both bad
        confidence_score < CONFIDENCE_THRESHOLD or  # Overall terrible
        (not source_scores and has_uncertainty)  # No sources + uncertain
    )
    
    # Create detailed reason message
    if has_uncertainty and retrieval_score < MIN_RETRIEVAL_SCORE:
        reason = "AI uncertain AND poor document retrieval"
    elif has_uncertainty:
        reason = "AI expressed uncertainty but sources available"
    elif retrieval_score < MIN_RETRIEVAL_SCORE:
        reason = "Very low quality document retrieval"
    elif retrieval_score < 0.35:
        reason = "Low quality document retrieval"
    elif not source_scores:
        reason = "No relevant documents found"
    else:
        reason = "Confident answer with good sources"
    
    return {
        'confidence_score': round(confidence_score, 2),
        'should_escalate': should_escalate,
        'reason': reason
    }
# ============================================================================
# PROCESS QUERY WITH SIMPLE ESCALATION
# ============================================================================

def process_query(user_question, llm_model, lang, mode=DEFAULT_QUERY_MODE):
    start_time = time.time()
    temperature = 0.1 if mode == "rag_only" else 0.4 if mode == "hybrid" else 0.7

    translations = get_translations(lang)

    # System prompts
    if lang == "FR":
        intro = "Vous êtes un assistant expert qui aide les utilisateurs à comprendre de la documentation technique."
        instruction = "Fournissez une réponse détaillée et pratique. Si vous ne trouvez pas l'information, dites-le clairement."
    else:
        intro = "You are an expert assistant helping users understand technical documentation."
        instruction = "Provide a detailed and practical answer. If you cannot find the information, state this clearly."

    # LLM-only mode
    if mode == "llm_only":
        prompt = f"{intro}\n\nQuestion: {user_question}\n\n{instruction}"
        answer = call_ollama_llm(prompt, llm_model, temperature=temperature)
        duration = time.time() - start_time
        
        # Check confidence for LLM-only
        confidence_info = check_answer_confidence(answer, [], lang)
        
        # If low confidence, return escalation message
        if confidence_info['should_escalate']:
            answer = translations["escalation_message"]
        
        return answer, [], duration, confidence_info

    # RAG mode: retrieve documents
    query_emb = embed_texts([user_question])[0]

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    scores = results.get("distances", [[]])[0]

    # Filter by threshold
    relevant_data = [
        (doc, meta, score)
        for doc, meta, score in zip(docs, metas, scores)
        if score >= THRESHOLD_GOOD
    ]

    # If no relevant documents found, escalate immediately
    if not relevant_data:
        duration = time.time() - start_time
        confidence_info = {
            'confidence_score': 0.0,
            'should_escalate': True,
            'reason': 'No relevant documents found'
        }
        escalation_msg = f"{translations['no_relevant_docs']}\n\n{translations['escalation_message']}"
        return escalation_msg, [], duration, confidence_info

    # Keep top N most relevant
    relevant_data = sorted(relevant_data, key=lambda x: x[2], reverse=True)[:TOP_K_RELEVANT]

    # Group by page
    page_map = {}
    for doc, meta, score in relevant_data:
        url = meta.get("url", "")
        if url not in page_map:
            page_map[url] = {"text": doc, "score": round(score, 4)}
        else:
            page_map[url]["text"] += "\n" + doc

    page_contexts = [
        {"url": url, "text": data["text"], "score": data["score"]}
        for url, data in page_map.items()
    ]

    # Build context
    all_text = "\n\n".join(p["text"] for p in page_contexts)[:MAX_CHARS]
    all_urls = [p["url"] for p in page_contexts]

    # Create prompt
    prompt = f"""
{intro}

Documentation sources:
{chr(10).join('- ' + url for url in all_urls)}

Documentation content:
{all_text}

User question: {user_question}

{instruction}

IMPORTANT: If the documentation doesn't contain the answer, say so clearly. Do not make up information.
"""

    # Get answer from LLM
    answer = call_ollama_llm(prompt, llm_model, temperature=temperature)
    duration = time.time() - start_time
    
    # Calculate confidence
    source_scores = [p["score"] for p in page_contexts]
    confidence_info = check_answer_confidence(answer, source_scores, lang)
    
    # If low confidence, escalate to human
    if confidence_info['should_escalate']:
        # Provide the AI's attempt + escalation message
        escalation_msg = f"{answer}\n\n---\n\n{translations['escalation_message']}"
        return escalation_msg, page_contexts, duration, confidence_info
    
    return answer, page_contexts, duration, confidence_info


# --- PDF Generation ---
def generate_pdf(content, lang):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
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
        pisa_status = pisa.CreatePDF(html_template, dest=tmp_file)
        tmp_file.seek(0)
        pdf_data = tmp_file.read()
    return html.A(
        get_translations(lang)["download_pdf"],
        href=f"data:application/pdf;base64,{base64.b64encode(pdf_data).decode('utf-8')}",
        download="rag_answer.pdf",
        target="_blank",
        className="btn btn-outline-info mt-3"
    )

# --- Dash App Setup ---
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "RAG Assistant with Escalation"

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("🤖 RAG Assistant", className="text-primary"), width=8),
        dbc.Col(html.Img(src="/assets/logo_centrale.svg", height="60px"), width=4, style={"textAlign": "right"})
    ], align="center"),

    html.Hr(),

    dbc.Row([
        dbc.Col(dbc.Textarea(
            id="question-input", 
            placeholder="Ask your question...", 
            style={"height": "120px"}, 
            className="mb-2"
        ), width=8),
        dbc.Col([
            html.Label("Model", className="text-info fw-bold"),
            dbc.Select(
                id="llm-selector",
                options=[
                    {"label": "Gemma 3 (4b)", "value": "gemma3:4b"},
                    {"label": "Mistral 7B", "value": "mistral:7b"}
                ],
                value=DEFAULT_LLM_MODEL,
                className="mb-3"
            ),
            html.Label("Language", className="text-info fw-bold"),
            dbc.Select(
                id="lang-selector",
                options=[
                    {"label": "English", "value": "EN"},
                    {"label": "Français", "value": "FR"}
                ],
                value=DEFAULT_LANGUAGE,
                className="mb-3"
            ),
            html.Label("Mode", className="text-info fw-bold"),
            dbc.Select(
                id="mode-selector",
                options=[
                    {"label": "Hybrid (RAG + LLM)", "value": "hybrid"},
                    {"label": "RAG Only", "value": "rag_only"},
                    {"label": "LLM Only", "value": "llm_only"}
                ],
                value=DEFAULT_QUERY_MODE,
                className="mb-3"
            )
        ], width=4)
    ]),

    dbc.Row([
        dbc.Col(dbc.Button("Submit", id="submit-button", color="success"), width="auto"),
        dbc.Col(dbc.Button("🧹 Clear", id="clear-button", color="warning", className="ms-2"), width="auto"),
        dbc.Col(dbc.Checkbox(id="show-sources-toggle", value=True, className="ms-3"), width="auto"),
        dbc.Col(html.Label("Show sources", className="mt-2"), width="auto")
    ], className="my-3", align="center"),

    dcc.Loading(
        id="loading-output",
        type="circle",
        color="#00ff99",
        children=[
            dbc.Card([
                html.Div(id="chat-history", children=[], style={"margin": "10px"})
            ], color="dark", inverse=True),
            html.Div(id="pdf-download", className="mt-3 text-end")
        ]
    )
], fluid=True, className="p-4", style={"backgroundColor": "#1e1e1e"})

def get_confidence_badge(confidence_score, should_escalate, lang):
    """Generate a simple confidence badge"""
    translations = get_translations(lang)
    
    if should_escalate:
        color = "danger"
        icon = "🔄"
        label = translations["low_confidence"]
    elif confidence_score < 0.75:
        color = "warning"
        icon = "⚠️"
        label = translations["medium_confidence"]
    else:
        color = "success"
        icon = "✅"
        label = translations["high_confidence"]
    
    return dbc.Badge(
        f"{icon} {label} ({confidence_score:.0%})",
        color=color,
        className="ms-2"
    )

@app.callback(
    Output("chat-history", "children"),
    Output("pdf-download", "children"),
    Output("question-input", "value"),
    Input("submit-button", "n_clicks"),
    Input("clear-button", "n_clicks"),
    State("question-input", "value"),
    State("show-sources-toggle", "value"),
    State("llm-selector", "value"),
    State("lang-selector", "value"),
    State("mode-selector", "value"),
    State("chat-history", "children"),
    prevent_initial_call=True
)
def update_chat(submit_clicks, clear_clicks, question, show_sources, llm_model, lang, mode, history):
    triggered_id = ctx.triggered_id
    
    # Clear button pressed
    if triggered_id == "clear-button":
        return [], "", ""

    # No question entered
    if not question or not question.strip():
        translations = get_translations(lang)
        return history + [html.Div(translations["no_question"])], "", question

    # Process the query
    translations = get_translations(lang)
    answer, source_data, latency, confidence_info = process_query(question, llm_model, lang, mode)
    
    # Format the answer
    formatted_answer = dcc.Markdown(answer)
    
    # Create confidence badge
    confidence_badge = get_confidence_badge(
        confidence_info['confidence_score'],
        confidence_info['should_escalate'],
        lang
    )
    
    # Show latency and confidence
    latency_info = html.Div([
        html.Span(f"⏱️ {latency:.2f}s", className="text-muted me-3"),
        confidence_badge
    ], style={"fontSize": "0.85em", "marginTop": "5px"})

    # Format sources
    source_block = ""
    if show_sources and source_data:
        sorted_sources = sorted(source_data, key=lambda x: x["score"], reverse=True)
        source_block = dcc.Markdown(
            "\n".join([f"- {item['url']} (score: {item['score']})" for item in sorted_sources])
        )

    # Generate PDF content
    pdf_sources = "\n".join([
        f"- {item['url']} (score: {item['score']})" 
        for item in source_data
    ]) if source_data else "No sources"
    
    pdf_content = f"""You: {question}

Answer:
{answer}

Confidence: {confidence_info['confidence_score']:.0%}
Reason: {confidence_info['reason']}

Sources:
{pdf_sources}"""

    download_link = generate_pdf(pdf_content, lang)

    # Create the chat exchange UI
    new_exchange = html.Div([
        html.H5(f"{translations['you']}:", className="text-warning"),
        html.Div(question, style={
            "whiteSpace": "pre-wrap", 
            "marginBottom": "10px"
        }),
        
        html.H5(f"{translations['assistant']}:", className="text-success"),
        html.Div([formatted_answer, latency_info], style={
            "backgroundColor": "#2a2a2a",
            "padding": "10px",
            "borderRadius": "10px",
            "marginBottom": "10px"
        }),
        
        html.Div([
            html.Strong(translations["sources_used"]), 
            source_block
        ], style={
            "marginTop": "10px",
            "color": "#ccc",
            "fontSize": "0.85em",
            "backgroundColor": "#1e1e1e",
            "padding": "8px",
            "borderRadius": "6px"
        }) if source_block else html.Div()
        
    ], style={"marginBottom": "30px"})

    return [new_exchange] + history, download_link, ""

if __name__ == "__main__":
    app.run(debug=True)