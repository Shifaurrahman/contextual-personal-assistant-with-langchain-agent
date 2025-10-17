#!/usr/bin/env python3
"""
app.py - Enhanced with True LangChain Agents

FastAPI backend for Contextual Personal Assistant with dual LangChain agents:
1. Ingestion Agent -  agent with tools for entity extraction, envelope fetching, and similarity computation
2. Thinking Agent -  agent with tools for insight computations (conflicts, duplicates, etc.)

Setup:
  1. Create a .env file with: OPENAI_API_KEY=sk-your-key-here
  2. pip install -r requirements.txt
  3. python -m spacy download en_core_web_sm
  
Run:
  uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

import os
import sqlite3
import json
import re
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, date, timedelta

import numpy as np
import dateparser
import spacy
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Modern LangChain imports for true agents
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain import hub
from langchain_core.tools import tool

# Additional imports for fixing proxies issue
from openai import OpenAI
import httpx

# Load environment variables
load_dotenv()

# ==================== Configuration ====================
DB_PATH = "assistant.db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
ENVELOPE_SIMILARITY_THRESHOLD = 0.50

# ==================== Load Models ====================
print("Loading models...")
try:
    nlp = spacy.load("en_core_web_sm")
    print("[OK] spaCy model loaded")
except Exception as e:
    raise RuntimeError("Install spaCy model: python -m spacy download en_core_web_sm") from e

try:
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"[OK] Sentence transformer loaded: {EMBEDDING_MODEL_NAME}")
except Exception as e:
    raise RuntimeError(f"Failed to load embedding model {EMBEDDING_MODEL_NAME}") from e

# ==================== Pydantic Models ====================
class NoteRequest(BaseModel):
    note: str

class CardResponse(BaseModel):
    card_id: int
    created_new_envelope: bool
    envelope_id: int
    envelope_score: float
    card_type: str
    extracted_date: Optional[str] = None
    extracted_time: Optional[str] = None

class Envelope(BaseModel):
    id: int
    name: str
    topic_keywords: List[str]
    created_at: str
    card_count: Optional[int] = 0

class Card(BaseModel):
    id: int
    type: str
    description: str
    date: Optional[str] = None
    time: Optional[str] = None
    assignee: Optional[str] = None
    context_keywords: List[str]
    envelope_id: Optional[int] = None
    created_at: str

class UserContext(BaseModel):
    active_projects: List[str]
    contacts: List[str]
    upcoming_deadlines: List[str]
    themes: List[str]

class ThinkingResponse(BaseModel):
    insights: Dict[str, Any]
    natural_text: str

# ==================== Database Functions ====================
def init_db(path: str = DB_PATH):
    """Initialize SQLite database with required tables."""
    con = sqlite3.connect(path)
    cur = con.cursor()
    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS cards (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT NOT NULL,
        description TEXT NOT NULL,
        date TEXT,
        time TEXT,
        assignee TEXT,
        context_keywords TEXT,
        envelope_id INTEGER,
        embedding BLOB,
        created_at TEXT NOT NULL
    )
    """)
    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS envelopes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        topic_keywords TEXT,
        embedding BLOB,
        created_at TEXT NOT NULL
    )
    """)
    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_context (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        context_json TEXT NOT NULL
    )
    """)
    
    initial_context = {
        "active_projects": [],
        "contacts": [],
        "upcoming_deadlines": [],
        "themes": []
    }
    cur.execute(
        "INSERT OR IGNORE INTO user_context (id, context_json) VALUES (1, ?)",
        (json.dumps(initial_context),)
    )
    
    con.commit()
    con.close()
    print("[OK] Database initialized")

def to_bytes(np_array: np.ndarray) -> bytes:
    return np_array.tobytes()

def from_bytes(b: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(b, dtype=dtype)

# ==================== NLP Processing Functions ====================
def preprocess_text(text: str) -> str:
    """Clean and normalize input text."""
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty")
    return " ".join(text.strip().split())

def extract_time(text: str) -> Optional[str]:
    """Extract time information from text, preserving am/pm indicator."""
    patterns = [
        r'\b(\d{1,2})\.(\d{2})\s*(a\.m\.|p\.m\.|AM|PM|am|pm)\b',
        r'\b(\d{1,2}):(\d{2})\s*(a\.m\.|p\.m\.|AM|PM|am|pm)\b',
        r'\b(\d{1,2})\s*(a\.m\.|p\.m\.|AM|PM|am|pm)\b',
        r'\b(\d{1,2}):(\d{2})(?!\s*(?:a\.m\.|p\.m\.|AM|PM|am|pm))\b',
        r'\b(\d{1,2})\.(\d{2})(?!\s*(?:a\.m\.|p\.m\.|AM|PM|am|pm))\b',
    ]
    
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            time_str = m.group(0)
            has_ampm = any(indicator in time_str.lower() for indicator in ['am', 'pm', 'a.m', 'p.m'])
            
            if has_ampm:
                if 'AM' in time_str:
                    time_str = time_str.replace('AM', 'a.m')
                elif 'am' in time_str and 'a.m' not in time_str:
                    time_str = time_str.replace('am', 'a.m')
                elif 'PM' in time_str:
                    time_str = time_str.replace('PM', 'p.m')
                elif 'pm' in time_str and 'p.m' not in time_str:
                    time_str = time_str.replace('pm', 'p.m')
                return time_str
            else:
                parts = time_str.replace(':', '.').split('.')
                if len(parts) == 2 and len(parts[1]) == 2:
                    return time_str
    
    return None

def parse_relative_date(text: str, base_date: date) -> Optional[date]:
    """Parse relative dates like 'tomorrow', 'next Monday', etc."""
    lower = text.lower()
    
    if any(phrase in lower for phrase in ['today', 'this evening', 'tonight']):
        return base_date
    
    if 'tomorrow' in lower:
        return base_date + timedelta(days=1)
    
    if 'next week' in lower:
        return base_date + timedelta(days=7)
    
    days_map = {
        'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
        'friday': 4, 'saturday': 5, 'sunday': 6
    }
    
    for day_name, day_num in days_map.items():
        if day_name in lower:
            current_weekday = base_date.weekday()
            
            if 'next' in lower:
                days_ahead = (day_num - current_weekday) % 7
                if days_ahead == 0:
                    days_ahead = 7
            else:
                days_ahead = (day_num - current_weekday) % 7
                if days_ahead == 0:
                    days_ahead = 7
            
            return base_date + timedelta(days=days_ahead)
    
    return None

def detect_card_type(text: str, has_temporal: bool) -> str:
    """Classify card type based on content."""
    doc = nlp(text)
    text_lower = text.lower()
    
    reminder_keywords = {"remember", "remind", "don't forget", "pickup", "pick up"}
    if any(k in text_lower for k in reminder_keywords):
        return "Reminder"
    
    action_verbs = {
        "call", "email", "schedule", "meet", "buy", "research",
        "create", "send", "finish", "review", "write", "prepare"
    }
    for tok in doc:
        if tok.pos_ == "VERB" and tok.lemma_.lower() in action_verbs:
            return "Task"
    
    if has_temporal:
        return "Reminder"
    
    return "Idea/Note"

def extract_entities(text: str) -> Tuple[Optional[str], Optional[str], Optional[str], List[str], bool]:
    """Extract entities from text using spaCy NER."""
    doc = nlp(text)
    base_date = date.today()
    
    assignee = None
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            if ent.start > 0:
                prev_token = doc[ent.start - 1]
                action_verbs = {'call', 'email', 'meet', 'contact', 'message', 'text'}
                if prev_token.lemma_.lower() in action_verbs:
                    assignee = ent.text
                else:
                    assignee = ent.text
            else:
                assignee = ent.text
            break
    
    if not assignee:
        for ent in doc.ents:
            if ent.label_ == "ORG":
                assignee = ent.text
                break
    
    has_temporal = any(ent.label_ in ("DATE", "TIME") for ent in doc.ents)
    
    time_str = extract_time(text)
    if time_str:
        has_temporal = True
    
    settings = {
        'PREFER_DATES_FROM': 'future',
        'RELATIVE_BASE': datetime.combine(base_date, datetime.min.time())
    }
    
    parsed = dateparser.parse(text, settings=settings)
    date_iso = None
    
    if parsed:
        date_iso = parsed.date().isoformat()
        has_temporal = True
    else:
        date_entities = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
        for date_text in date_entities:
            parsed = dateparser.parse(date_text, settings=settings)
            if parsed:
                date_iso = parsed.date().isoformat()
                has_temporal = True
                break
        
        if not date_iso:
            rel = parse_relative_date(text, base_date)
            if rel:
                date_iso = rel.isoformat()
                has_temporal = True
    
    keywords = set()
    temporal_keywords = {
        'today', 'tomorrow', 'tonight', 'week', 'month',
        'next', 'this', 'last', 'yesterday', 'friday', 'monday',
        'tuesday', 'wednesday', 'thursday', 'saturday', 'sunday'
    }
    
    for ent in doc.ents:
        if ent.label_ not in ("DATE", "TIME") and len(ent.text) > 2:
            k = ent.text.lower()
            if k not in temporal_keywords:
                keywords.add(k)
    
    for tok in doc:
        if tok.pos_ in ("NOUN", "PROPN") and not tok.is_stop and len(tok.lemma_) > 2:
            k = tok.lemma_.lower()
            if k not in temporal_keywords:
                keywords.add(k)
    
    return date_iso, time_str, assignee, sorted(list(keywords)), has_temporal

def compute_embedding(text: str) -> np.ndarray:
    """Generate semantic embedding for text."""
    emb = embed_model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return emb.astype(np.float32)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))

# Helper to serialize embedding for JSON (list of floats)
def embedding_to_list(emb: np.ndarray) -> list:
    return emb.tolist()

def list_to_embedding(emb_list: list) -> np.ndarray:
    return np.array(emb_list, dtype=np.float32)

# ==================== Database Operations ====================
def insert_card(card: Dict[str, Any]) -> int:
    """Insert a new card into the database."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
    INSERT INTO cards (type, description, date, time, assignee, 
                      context_keywords, envelope_id, embedding, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        card.get("type"),
        card.get("description"),
        card.get("date"),
        card.get("time"),
        card.get("assignee"),
        json.dumps(card.get("context_keywords", [])),
        card.get("envelope_id"),
        to_bytes(card["embedding"]),
        datetime.utcnow().isoformat()
    ))
    con.commit()
    rowid = cur.lastrowid
    con.close()
    return rowid

def get_envelopes() -> List[Dict[str, Any]]:
    """Retrieve all envelopes from database."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT id, name, topic_keywords, embedding, created_at FROM envelopes")
    rows = cur.fetchall()
    
    envs = []
    for r in rows:
        emb = from_bytes(r[3]) if r[3] else None
        envs.append({
            "id": r[0],
            "name": r[1],
            "topic_keywords": json.loads(r[2]) if r[2] else [],
            "embedding": embedding_to_list(emb) if emb is not None else None,  # Serialize for JSON
            "created_at": r[4]
        })
    con.close()
    return envs

def insert_envelope(name: str, topic_keywords: List[str], embedding: np.ndarray) -> int:
    """Create a new envelope."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO envelopes (name, topic_keywords, embedding, created_at) VALUES (?, ?, ?, ?)",
        (name, json.dumps(topic_keywords), to_bytes(embedding), datetime.utcnow().isoformat())
    )
    con.commit()
    nid = cur.lastrowid
    con.close()
    return nid

def get_cards(envelope_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Retrieve cards, optionally filtered by envelope."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    
    if envelope_id is None:
        cur.execute("""
        SELECT id, type, description, date, time, assignee, 
               context_keywords, envelope_id, created_at 
        FROM cards ORDER BY created_at DESC
        """)
    else:
        cur.execute("""
        SELECT id, type, description, date, time, assignee, 
               context_keywords, envelope_id, created_at 
        FROM cards WHERE envelope_id = ? ORDER BY created_at DESC
        """, (envelope_id,))
    
    rows = cur.fetchall()
    cards = []
    for r in rows:
        cards.append({
            "id": r[0],
            "type": r[1],
            "description": r[2],
            "date": r[3],
            "time": r[4],
            "assignee": r[5],
            "context_keywords": json.loads(r[6]) if r[6] else [],
            "envelope_id": r[7],
            "created_at": r[8]
        })
    con.close()
    return cards

def get_user_context() -> Dict[str, Any]:
    """Retrieve current user context."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT context_json FROM user_context WHERE id=1")
    row = cur.fetchone()
    con.close()
    
    if row:
        return json.loads(row[0])
    return {
        "active_projects": [],
        "contacts": [],
        "upcoming_deadlines": [],
        "themes": []
    }

def update_user_context(updates: Dict[str, Any]):
    """Update user context with new information."""
    ctx = get_user_context()
    
    for k, v in updates.items():
        if isinstance(v, list):
            existing = ctx.get(k, [])
            seen = set(existing)
            for it in v:
                if it not in seen:
                    existing.append(it)
                    seen.add(it)
            ctx[k] = existing
        else:
            ctx[k] = v
    
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "UPDATE user_context SET context_json = ? WHERE id = 1",
        (json.dumps(ctx),)
    )
    con.commit()
    con.close()

# ==================== Tools for Ingestion Agent ====================
@tool
def perform_base_extraction(text: str) -> str:
    """Perform base entity extraction using spaCy and return as JSON string."""
    text_clean = preprocess_text(text)
    date_iso, time_str, assignee, keywords, has_temporal = extract_entities(text_clean)
    card_type = detect_card_type(text_clean, has_temporal)
    base_extraction = {
        "card_type": card_type,
        "description": text_clean,
        "date": date_iso,
        "time": time_str,
        "assignee": assignee,
        "keywords": keywords
    }
    return json.dumps(base_extraction)

@tool
def fetch_envelopes() -> str:
    """Fetch all envelopes from the database and return as JSON string."""
    envelopes = get_envelopes()
    return json.dumps(envelopes)

@tool
def compute_card_embedding(text: str) -> str:
    """Compute semantic embedding for the given text and return as JSON list."""
    emb = compute_embedding(text)
    return json.dumps(embedding_to_list(emb))

@tool
def compute_envelope_similarity(card_emb_list: str, env_emb_list: str) -> float:
    """Compute cosine similarity between card embedding and envelope embedding."""
    card_emb = list_to_embedding(json.loads(card_emb_list))
    env_emb = list_to_embedding(json.loads(env_emb_list))
    return cosine_similarity(card_emb, env_emb)

@tool
def create_new_envelope(name: str, keywords: List[str]) -> int:
    """Create a new envelope with the given name and keywords, return the ID."""
    emb = compute_embedding(" ".join(keywords))
    return insert_envelope(name, keywords, emb)

# ==================== LANGCHAIN AGENT 1: INGESTION AGENT ( AGENT) ====================
class IngestionAgent:
    """True LangChain agent for enhanced entity extraction and classification using tools."""
    
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found in environment")
        
        # Create ChatOpenAI with explicit configuration
        self.llm = ChatOpenAI(
            temperature=0.1,
            model="gpt-4o-mini",
            openai_api_key=api_key,
            http_client=httpx.Client(timeout=30.0)
        )
        # Define tools
        self.tools = [
            perform_base_extraction,
            fetch_envelopes,
            compute_card_embedding,
            compute_envelope_similarity,
            create_new_envelope
        ]
        
        # Pull a standard agent prompt from hub
        prompt = hub.pull("hwchase17/openai-functions-agent")
        prompt.messages[0].prompt.template = """You are an expert at extracting and organizing information from notes into cards and envelopes.

Use the available tools to:
1. Extract base entities from the note text.
2. Enhance or correct the extraction based on rules (e.g., fix assignee, remove temporals from keywords).
3. Fetch existing envelopes and suggest the best match using similarity.
4. If no good match, create a new envelope.

For enhancement, return JSON: {{"card_type": "...", "description": "...", "date": "...", "time": "...", "assignee": "...", "context_keywords": [...]}}
For envelope suggestion, return JSON: {{"envelope_id": <int or null>, "confidence": <float>, "reasoning": "..."}} 

Today: {today}
Tomorrow: {tomorrow}

{input}
{agent_scratchpad}"""
        
        self.agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        self.executor = AgentExecutor(
            agent=self.agent, 
            tools=self.tools, 
            verbose=True,  # For debugging, set to False in production
            handle_parsing_errors=True
        )
        print("[OK] Ingestion Agent initialized as true agent with tools")
    
    def enhance_extraction(self, text: str, base_extraction: Dict[str, Any]) -> Dict[str, Any]:
        """Use the agent to enhance or correct the base spaCy extraction."""
        today = date.today().isoformat()
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        
        input_msg = f"""Enhance the extraction for this note: {text}

Preliminary extraction: {json.dumps(base_extraction)}

Refine based on rules: 
- If action like "Call Sarah", assignee="Sarah"
- Remove temporal words from keywords
- Keep 2-5 meaningful keywords
- Card types: Task|Reminder|Idea/Note

Return ONLY valid JSON for the enhanced extraction."""

        try:
            result = self.executor.invoke({
                "input": input_msg,
                "today": today,
                "tomorrow": tomorrow
            })
            # Parse the agent's output for JSON
            json_match = re.search(r'\{.*\}', result['output'], re.DOTALL)
            if json_match:
                enhanced = json.loads(json_match.group())
                print(f"[Agent Enhanced] {enhanced}")
                return enhanced
            else:
                return base_extraction
        except Exception as e:
            print(f"[WARNING] Agent enhancement failed: {e}, using base extraction")
            return base_extraction
    
    def suggest_envelope(self, card_data: Dict[str, Any], envelopes: List[Dict]) -> Tuple[Optional[int], str]:
        """Use the agent to suggest best envelope using tools."""
        today = date.today().isoformat()
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        input_msg = f"""Suggest envelope for this card: {json.dumps(card_data)}

Available envelopes: {json.dumps(envelopes)}

Use tools to compute embeddings and similarities if needed.
Rules:
- Match on keyword overlap (2+ common) or semantic similarity >0.6
- Prefer assignee match
- If low confidence, return null to create new

Return ONLY valid JSON: {{"envelope_id": <int or null>, "confidence": <0.0-1.0>, "reasoning": "brief"}}"""

        try:
            result = self.executor.invoke({
                "input": input_msg,
                "today": today,
                "tomorrow": tomorrow
            })
            json_match = re.search(r'\{.*\}', result['output'], re.DOTALL)
            if json_match:
                suggestion = json.loads(json_match.group())
                env_id = suggestion.get("envelope_id")
                confidence = suggestion.get("confidence", 0.5)
                reasoning = suggestion.get("reasoning", "")
                
                print(f"[Agent Envelope] ID={env_id}, Conf={confidence}, Reason={reasoning}")
                
                if env_id and confidence >= 0.6:
                    if any(e["id"] == env_id for e in envelopes):
                        return env_id, reasoning
                
                return None, reasoning
            else:
                return None, "Failed to parse agent response"
        except Exception as e:
            print(f"[WARNING] Agent envelope suggestion failed: {e}")
            return None, str(e)

_ingestion_agent = None

def get_ingestion_agent():
    """Get or create ingestion agent."""
    global _ingestion_agent
    if _ingestion_agent is None:
        try:
            _ingestion_agent = IngestionAgent()
        except Exception as e:
            print(f"[WARNING] Failed to initialize IngestionAgent: {e}")
            _ingestion_agent = None
    return _ingestion_agent

# ==================== Tools for Thinking Agent ====================
@tool
def fetch_all_cards() -> str:
    """Fetch all cards from the database and return as JSON string."""
    cards = get_cards()
    return json.dumps(cards)

@tool
def fetch_cards_by_type(types: List[str]) -> str:
    """Fetch cards filtered by types (e.g., ['Task', 'Reminder']) and return as JSON."""
    filtered = [c for c in get_cards() if c["type"] in types]
    return json.dumps(filtered)

@tool
def detect_scheduling_conflicts() -> str:
    """Detect scheduling conflicts and return as JSON list."""
    cards = get_cards()
    today = date.today()
    tasks = [c for c in cards if c["type"] in ("Task", "Reminder") 
             and c["date"] and c.get("assignee")]
    
    date_assignee_map = {}
    for t in tasks:
        key = (t["date"], t["assignee"])
        date_assignee_map.setdefault(key, []).append(t)
    
    conflicts = []
    for k, items in date_assignee_map.items():
        if len(items) > 1:
            conflicts.append({
                "date": k[0],
                "assignee": k[1],
                "tasks": [{"id": it["id"], "description": it["description"]} 
                         for it in items]
            })
    return json.dumps(conflicts)

@tool
def suggest_envelope_merges() -> str:
    """Suggest envelope merges based on keyword overlap and return as JSON."""
    envelopes = get_envelopes()  # Deserialized embeddings not needed here
    suggestions = []
    for i, e1 in enumerate(envelopes):
        for j in range(i + 1, len(envelopes)):
            e2 = envelopes[j]
            set1 = set([str(x).lower() for x in e1.get("topic_keywords", [])])
            set2 = set([str(x).lower() for x in e2.get("topic_keywords", [])])
            
            if not set1 or not set2:
                continue
            
            inter = set1.intersection(set2)
            overlap = len(inter) / min(len(set1), len(set2))
            
            if overlap >= 0.5:
                suggestions.append({
                    "envelope_a": {"id": e1["id"], "name": e1["name"]},
                    "envelope_b": {"id": e2["id"], "name": e2["name"]},
                    "overlap": round(overlap, 2),
                    "common_keywords": list(inter)[:5]
                })
    return json.dumps(suggestions)

@tool
def find_next_steps() -> str:
    """Find next steps suggestions for envelopes and return as JSON."""
    envelopes = get_envelopes()
    suggestions = []
    for env in envelopes:
        env_cards = get_cards(env["id"])
        t_env = [c for c in env_cards if c["type"] in ("Task", "Reminder")]
        ideas = [c for c in env_cards if c["type"] == "Idea/Note"]
        
        if len(t_env) > 0 and len(ideas) == 0:
            suggestions.append({
                "envelope": {"id": env["id"], "name": env["name"]},
                "suggestion": f"Consider documenting learnings or planning next phase for '{env['name']}'"
            })
    return json.dumps(suggestions)

@tool
def find_overdue_tasks() -> str:
    """Find overdue tasks and return as JSON."""
    cards = get_cards()
    today = date.today()
    overdue = []
    for card in cards:
        if card["type"] in ("Task", "Reminder") and card.get("date"):
            try:
                cd = date.fromisoformat(card["date"])
                if cd < today:
                    overdue.append({
                        "id": card["id"],
                        "description": card["description"],
                        "due_date": card["date"],
                        "days_overdue": (today - cd).days,
                        "assignee": card.get("assignee")
                    })
            except:
                pass
    overdue.sort(key=lambda x: x["days_overdue"], reverse=True)
    return json.dumps(overdue)

@tool
def find_orphaned_contacts() -> str:
    """Find orphaned contacts and return as JSON."""
    cards = get_cards()
    all_assignees = set()
    recent = set()
    
    for card in cards:
        if card.get("assignee"):
            all_assignees.add(card["assignee"])
            try:
                created = datetime.fromisoformat(card["created_at"])
                if (datetime.utcnow() - created).days <= 7:
                    recent.add(card["assignee"])
            except:
                pass
    
    orphaned = all_assignees - recent
    results = []
    for a in orphaned:
        acards = [c for c in cards if c.get("assignee") == a]
        results.append({
            "assignee": a,
            "last_mention": max([c["created_at"] for c in acards]),
            "total_cards": len(acards)
        })
    return json.dumps(results)

@tool
def find_priority_tasks() -> str:
    """Find priority tasks (next 3 days) and return as JSON."""
    cards = get_cards()
    today = date.today()
    priority = []
    for card in cards:
        if card["type"] in ("Task", "Reminder") and card.get("date"):
            try:
                cd = date.fromisoformat(card["date"])
                days_until = (cd - today).days
                
                if 0 <= days_until <= 3:
                    priority.append({
                        "id": card["id"],
                        "description": card["description"],
                        "due_date": card["date"],
                        "days_until": days_until,
                        "assignee": card.get("assignee"),
                        "urgency": "High" if days_until == 0 else "Medium"
                    })
            except:
                pass
    priority.sort(key=lambda x: x["days_until"])
    return json.dumps(priority)

@tool
def detect_potential_duplicates() -> str:
    """Detect potential duplicate tasks and return as JSON."""
    from difflib import SequenceMatcher
    cards = get_cards()
    duplicates = []
    for i, c1 in enumerate(cards):
        for j in range(i + 1, len(cards)):
            c2 = cards[j]
            sim = SequenceMatcher(
                None,
                c1["description"].lower(),
                c2["description"].lower()
            ).ratio()
            
            if sim >= 0.7:
                duplicates.append({
                    "card_a": {"id": c1["id"], "description": c1["description"]},
                    "card_b": {"id": c2["id"], "description": c2["description"]},
                    "similarity": round(sim, 2)
                })
    return json.dumps(duplicates)

# ==================== LANGCHAIN AGENT 2: THINKING AGENT ( AGENT) ====================
class ThinkingAgent:
    """True LangChain agent for generating insights using tools for computations."""
    
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found in environment")
        
        # Fix for proxies error: Explicitly create OpenAI client with no proxies
        http_client = httpx.Client(proxies=None, timeout=30.0)
        openai_client = OpenAI(
            api_key=api_key,
            http_client=http_client
        )
        
        self.llm = ChatOpenAI(
            temperature=0.2,
            model="gpt-4o-mini",
            client=openai_client
        )
        
        # Define tools for insight computations
        self.tools = [
            fetch_all_cards,
            fetch_cards_by_type,
            detect_scheduling_conflicts,
            suggest_envelope_merges,
            find_next_steps,
            find_overdue_tasks,
            find_orphaned_contacts,
            find_priority_tasks,
            detect_potential_duplicates
        ]
        
        # Pull standard agent prompt
        prompt = hub.pull("hwchase17/openai-functions-agent")
        prompt.messages[0].prompt.template = """You are an AI assistant that analyzes cards and generates structured insights.

Use tools to compute different insight categories:
- Conflicts, overdue, priority, duplicates, merges, next steps, orphaned.

Assemble all into a dict: {{"conflicts": [...], "overdue_tasks": [...], ...}}

Then, format into natural language with emojis and sections as per rules.

{input}
{agent_scratchpad}"""
        
        self.agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        self.executor = AgentExecutor(
            agent=self.agent, 
            tools=self.tools, 
            verbose=True,
            handle_parsing_errors=True
        )
        print("[OK] Thinking Agent initialized as true agent with tools")
    
    @staticmethod
    def _generate_fallback_summary(insights: dict) -> str:
        """Fallback if agent fails."""
        lines = ["🤔 Enhanced Thinking Agent Insights:\n"]
        
        # Priority Tasks
        if insights.get("priority_tasks"):
            lines.append("⚡ Priority Tasks (Next 3 Days):")
            for task in insights["priority_tasks"][:5]:
                urgency = task.get("urgency", "Medium")
                desc = task.get("description", "")
                due = task.get("due_date", "")
                days = task.get("days_until", 0)
                
                if days == 0:
                    due_text = "Due: TODAY"
                elif days == 1:
                    due_text = f"Due: tomorrow ({due})"
                else:
                    due_text = f"Due: {due}"
                
                lines.append(f"  • [{urgency}] {desc} {due_text}")
            lines.append("")
        
        # Scheduling Conflicts
        if insights.get("conflicts"):
            lines.append("⚠️ Scheduling Conflicts:")
            for conflict in insights["conflicts"][:3]:
                assignee = conflict.get("assignee", "Unknown")
                date = conflict.get("date", "")
                tasks = conflict.get("tasks", [])
                task_list = " - ".join([t.get("description", "")[:50] for t in tasks[:2]])
                lines.append(f"  • {assignee} on {date}: {task_list}")
            lines.append("")
        
        # Potential Duplicates
        if insights.get("potential_duplicates"):
            lines.append("🔄 Potential Duplicate Tasks:")
            shown = 0
            for dup in insights["potential_duplicates"]:
                if shown >= 5:
                    break
                sim = int(dup.get("similarity", 0) * 100)
                if sim >= 80:
                    card_a = dup.get("card_a", {}).get("description", "")[:50]
                    card_b = dup.get("card_b", {}).get("description", "")[:50]
                    lines.append(f"  • \"{card_a}\" vs \"{card_b}\" Similarity: {sim}%")
                    shown += 1
            if shown > 0:
                lines.append("")
        
        # Next Steps
        if insights.get("next_steps"):
            lines.append("💡 Suggested Next Steps:")
            for step in insights["next_steps"][:5]:
                env_name = step.get("envelope", {}).get("name", "")
                lines.append(f"  • Consider documenting learnings for '{env_name}'")
            lines.append("")
        
        # Overdue Tasks
        if insights.get("overdue_tasks"):
            lines.append("🔴 Overdue Tasks:")
            for task in insights["overdue_tasks"][:3]:
                desc = task.get("description", "")
                days = task.get("days_overdue", 0)
                lines.append(f"  • {desc} - {days} days overdue")
            lines.append("")
        
        return "\n".join(lines)
    
    def generate_insights(self, insights_dict: dict) -> str:
        """Use the agent to generate natural language from structured insights."""
        input_msg = f"""Given these computed insights: {json.dumps(insights_dict)}

Create a beautiful, actionable summary with emojis:
- Start: 🤔 Enhanced Thinking Agent Insights:
- Sections: ⚡ Priority Tasks, 🔴 Overdue, ⚠️ Conflicts, 🔄 Duplicates, 💡 Next Steps, 🔗 Merges
- Format priority: [Urgency] Desc Due: Date
- Duplicates: "Task1" vs "Task2" Sim: XX%
- Concise, only include non-empty sections"""

        try:
            result = self.executor.invoke({"input": input_msg})
            return result['output']
        except Exception as e:
            print(f"[ERROR] Agent summary generation failed: {e}")
            import traceback
            traceback.print_exc()
            return self._generate_fallback_summary(insights_dict)

_thinking_agent = None

def get_thinking_agent():
    """Get or create thinking agent."""
    global _thinking_agent
    if _thinking_agent is None:
        try:
            _thinking_agent = ThinkingAgent()
        except Exception as e:
            print(f"[WARNING] Failed to initialize ThinkingAgent: {e}")
            _thinking_agent = None
    return _thinking_agent

# ==================== Enhanced Processing Pipeline ====================
def process_note_with_llm(text: str) -> Dict[str, Any]:
    """Enhanced pipeline with True LangChain Ingestion Agent."""
    
    text_clean = preprocess_text(text)
    
    # Base extraction without agent first
    date_iso, time_str, assignee, keywords, has_temporal = extract_entities(text_clean)
    card_type = detect_card_type(text_clean, has_temporal)
    
    base_extraction = {
        "card_type": card_type,
        "description": text_clean,
        "date": date_iso,
        "time": time_str,
        "assignee": assignee,
        "keywords": keywords
    }
    
    agent = get_ingestion_agent()
    if agent:
        try:
            enhanced = agent.enhance_extraction(text_clean, base_extraction)
            
            card_type = enhanced.get("card_type", card_type)
            description = enhanced.get("description", text_clean)
            date_iso = enhanced.get("date", date_iso)
            time_str = enhanced.get("time", time_str)
            assignee = enhanced.get("assignee", assignee)
            keywords = enhanced.get("context_keywords", keywords)
            
            keywords = [k for k in keywords if k and len(k) > 1]
            
        except Exception as e:
            print(f"[INFO] Using base extraction (Agent unavailable): {e}")
            description = text_clean
            keywords = [k for k in keywords if k and len(k) > 1]
    else:
        print("[INFO] Using base extraction (Agent not initialized)")
        description = text_clean
        keywords = [k for k in keywords if k and len(k) > 1]
    
    emb_text = description + " " + " ".join(keywords)
    embedding = compute_embedding(emb_text)
    
    envelopes = get_envelopes()
    env_id = None
    score = 0.0
    created_new = False
    
    if not keywords or len(keywords) == 0:
        misc_env = next((e for e in envelopes if 'miscellaneous' in e['name'].lower()), None)
        if misc_env:
            env_id = misc_env['id']
            score = 0.5
            print(f"[Envelope] Using existing Miscellaneous (no keywords)")
        else:
            env_id = insert_envelope("Miscellaneous", ["general"], embedding)
            created_new = True
            score = 1.0
            print(f"[Envelope] Created Miscellaneous (no keywords)")
    else:
        if agent:
            try:
                suggested_id, reasoning = agent.suggest_envelope({
                    "description": description,
                    "context_keywords": keywords,
                    "assignee": assignee
                }, envelopes)
                
                if suggested_id and any(e["id"] == suggested_id for e in envelopes):
                    env_id = suggested_id
                    score = 0.8
                    print(f"[Agent Envelope] Assigned to {env_id}: {reasoning}")
                else:
                    print(f"[Agent Envelope] Suggests new envelope: {reasoning}")
                    env_id = None
                    score = 0.0
            except Exception as e:
                print(f"[INFO] Agent envelope suggestion unavailable: {e}")
                env_id = None
                score = 0.0
        else:
            print("[INFO] Agent envelope suggestion unavailable (not initialized)")
            env_id = None
            score = 0.0
        
        if env_id is None:
            best_id, best_score = find_best_envelope_semantic(embedding, keywords, envelopes)
            
            if best_score >= 0.65:
                env_id = best_id
                score = best_score
                env_name = next((e['name'] for e in envelopes if e['id'] == best_id), "Unknown")
                print(f"[Semantic] Assigned to {env_id} ({env_name}) with score {best_score:.2f}")
            else:
                name = generate_envelope_name_from_keywords(keywords)
                
                existing_names = [e['name'].lower() for e in envelopes]
                if name.lower() in existing_names:
                    if assignee:
                        name = f"{name} & {assignee}"
                    else:
                        name = f"{name} Tasks"
                
                env_id = insert_envelope(name, keywords, embedding)
                created_new = True
                score = 1.0
                print(f"[New Envelope] Created '{name}' with keywords: {keywords}")
    
    card = {
        "type": card_type,
        "description": description,
        "date": date_iso,
        "time": time_str,
        "assignee": assignee,
        "context_keywords": keywords,
        "envelope_id": env_id,
        "embedding": embedding
    }
    card_id = insert_card(card)
    
    context_updates = {}
    if assignee:
        context_updates["contacts"] = [assignee]
    if date_iso:
        context_updates["upcoming_deadlines"] = [date_iso]
    if keywords:
        context_updates["themes"] = keywords[:5]
        envs = get_envelopes()
        env_name = next((e["name"] for e in envs if e["id"] == env_id), None)
        if env_name and 'miscellaneous' not in env_name.lower():
            context_updates["active_projects"] = [env_name]
    
    if context_updates:
        update_user_context(context_updates)
    
    return {
        "card_id": card_id,
        "created_new_envelope": created_new,
        "envelope_id": env_id,
        "envelope_score": score,
        "card_type": card_type,
        "extracted_date": date_iso,
        "extracted_time": time_str
    }

def find_best_envelope_semantic(embedding: np.ndarray, keywords: List[str], envelopes: List[Dict]) -> Tuple[Optional[int], float]:
    """Find best envelope using semantic similarity (fallback)."""
    if not envelopes:
        return None, 0.0
    
    best_id = None
    best_score = 0.0
    
    for env in envelopes:
        env_emb_list = env.get("embedding")
        env_emb = list_to_embedding(env_emb_list) if env_emb_list else None
        emb_score = cosine_similarity(embedding, env_emb) if env_emb is not None else 0.0
        
        env_keywords = set([k.lower() for k in env.get("topic_keywords", [])])
        card_keywords = set([k.lower() for k in keywords])
        keyword_score = 0.0
        
        if env_keywords and card_keywords:
            intersection = len(env_keywords.intersection(card_keywords))
            union = max(len(env_keywords), len(card_keywords))
            keyword_score = intersection / union
        
        combined = 0.7 * emb_score + 0.3 * keyword_score
        
        if combined > best_score:
            best_score = combined
            best_id = env["id"]
    
    return best_id, best_score

def generate_envelope_name_from_keywords(keywords: List[str]) -> str:
    """Generate a meaningful envelope name from keywords."""
    if not keywords:
        return "Miscellaneous"
    
    temporal = {'today', 'tomorrow', 'next', 'this', 'week', 'month'}
    filtered = [k for k in keywords if k not in temporal]
    
    if not filtered:
        filtered = keywords
    
    meaningful = sorted(filtered, key=len, reverse=True)[:2]
    
    if not meaningful:
        meaningful = keywords[:2]
    
    return " & ".join([m.title() for m in meaningful])

# ==================== Thinking Agent Logic (Now Agent-Driven) ====================
def thinking_agent_run() -> Dict[str, Any]:
    """Run the thinking agent to generate insights using tools."""
    agent = get_thinking_agent()
    if agent:
        try:
            input_msg = "Compute all insight categories using tools: conflicts, merges, next_steps, overdue_tasks, orphaned_contacts, priority_tasks, potential_duplicates. Assemble into a single insights dict."
            
            result = agent.executor.invoke({"input": input_msg})
            
            # Parse output for insights dict
            json_match = re.search(r'\{.*\}', result['output'], re.DOTALL)
            if json_match:
                insights = json.loads(json_match.group())
            else:
                raise ValueError("No valid JSON in agent output")
            return insights
        except Exception as e:
            print(f"[WARNING] Agent insights computation failed: {e}, using fallback")
    
    # Fallback to original rule-based
    print("[INFO] Using fallback for insights computation")
    insights = {
        "conflicts": json.loads(detect_scheduling_conflicts()),
        "merge_suggestions": json.loads(suggest_envelope_merges()),
        "next_steps": json.loads(find_next_steps()),
        "overdue_tasks": json.loads(find_overdue_tasks()),
        "orphaned_contacts": json.loads(find_orphaned_contacts()),
        "priority_tasks": json.loads(find_priority_tasks()),
        "potential_duplicates": json.loads(detect_potential_duplicates())
    }
    
    return insights

# ==================== FastAPI Application ====================
app = FastAPI(
    title="Contextual Personal Assistant API",
    description="Backend API with true LangChain agents for intelligent note processing",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize database and agents on startup."""
    init_db()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("[OK] OpenAI API key loaded from .env")
        try:
            ingestion = get_ingestion_agent()
            thinking = get_thinking_agent()
            if ingestion:
                print("[OK] Ingestion Agent initialized successfully")
            if thinking:
                print("[OK] Thinking Agent initialized successfully")
            if not ingestion or not thinking:
                print("[WARNING] Some agents failed to initialize - falling back where possible")
        except Exception as e:
            print(f"[WARNING] Agent initialization failed: {e}")
            print("   Falling back to base extraction")
    else:
        print("[WARNING] OPENAI_API_KEY not found in .env file")
        print("   Agent functionality will not work without it")
    
    print("[OK] API server ready")

@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "message": "Contextual Personal Assistant API v2.1 with True LangChain Agents",
        "version": "2.1.0",
        "agents": {
            "ingestion": "True LangChain agent with tools for extraction & organization (GPT-4o-mini)",
            "thinking": "True LangChain agent with tools for insights computation (GPT-4o-mini)"
        },
        "endpoints": {
            "POST /add": "Add a new note (uses Ingestion Agent)",
            "GET /cards": "List all cards",
            "GET /envelopes": "List all envelopes",
            "GET /context": "Get user context",
            "GET /think": "Run thinking agent",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    api_key_status = "configured" if os.getenv("OPENAI_API_KEY") else "missing"
    
    agents_status = {
        "ingestion": "not_initialized" if _ingestion_agent is None else "ready",
        "thinking": "not_initialized" if _thinking_agent is None else "ready"
    }
    
    return {
        "status": "healthy" if api_key_status == "configured" and agents_status["ingestion"] == "ready" and agents_status["thinking"] == "ready" else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "openai_api_key": api_key_status,
        "agents": agents_status
    }

@app.post("/add", response_model=CardResponse)
async def api_add(request: NoteRequest):
    """Process a new note using True LangChain Ingestion Agent."""
    if not request.note.strip():
        raise HTTPException(status_code=400, detail="Note cannot be empty")
    
    try:
        result = process_note_with_llm(request.note)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing note: {str(e)}")

@app.get("/cards", response_model=List[Card])
async def api_list_cards(envelope_id: Optional[int] = Query(None, description="Filter by envelope ID")):
    """Get all cards, optionally filtered by envelope."""
    try:
        cards = get_cards(envelope_id)
        return cards
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching cards: {str(e)}")

@app.get("/envelopes", response_model=List[Envelope])
async def api_list_envelopes():
    """Get all envelopes with card counts."""
    try:
        envs = get_envelopes()
        for env in envs:
            env["card_count"] = len(get_cards(env["id"]))
        return envs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching envelopes: {str(e)}")

@app.get("/context", response_model=UserContext)
async def api_context():
    """Get current user context (projects, contacts, deadlines, themes)."""
    try:
        context = get_user_context()
        return context
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching context: {str(e)}")

@app.get("/think/debug")
async def api_think_debug():
    """Debug endpoint to see raw insights without LLM processing."""
    try:
        insights = thinking_agent_run()
        
        non_empty = {
            k: len(v) for k, v in insights.items() 
            if isinstance(v, list) and len(v) > 0
        }
        
        return {
            "insights": insights,
            "summary": {
                "total_categories": len(insights),
                "non_empty_categories": non_empty,
                "has_data": len(non_empty) > 0
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/think", response_model=ThinkingResponse)
async def api_think(natural: bool = Query(True, description="Return natural language summary")):
    """Run the thinking agent to analyze all cards and generate insights."""
    try:
        # Get structured insights via agent
        insights = thinking_agent_run()
        
        # Ensure serializable
        insights_clean = json.loads(json.dumps(insights, default=str))
        
        # Generate natural language summary if requested
        if natural:
            agent = get_thinking_agent()
            if agent:
                try:
                    natural_text = agent.generate_insights(insights_clean)
                except Exception as e:
                    print(f"[ERROR] Natural language generation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    natural_text = ThinkingAgent._generate_fallback_summary(insights_clean)
            else:
                print("[INFO] Using fallback summary (Agent not initialized)")
                natural_text = ThinkingAgent._generate_fallback_summary(insights_clean)
            
            return {
                "insights": insights_clean,
                "natural_text": natural_text
            }
        else:
            return {
                "insights": insights_clean,
                "natural_text": "Natural language generation disabled. Set natural=true to enable."
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running thinking agent: {str(e)}")

@app.delete("/cards/{card_id}")
async def api_delete_card(card_id: int):
    """Delete a specific card."""
    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute("DELETE FROM cards WHERE id = ?", (card_id,))
        con.commit()
        affected = cur.rowcount
        con.close()
        
        if affected == 0:
            raise HTTPException(status_code=404, detail="Card not found")
        
        return {"message": f"Card {card_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting card: {str(e)}")

@app.delete("/envelopes/{envelope_id}")
async def api_delete_envelope(envelope_id: int):
    """Delete an envelope and reassign its cards."""
    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        
        cur.execute("SELECT id FROM envelopes WHERE id = ?", (envelope_id,))
        if not cur.fetchone():
            con.close()
            raise HTTPException(status_code=404, detail="Envelope not found")
        
        cur.execute("SELECT id FROM envelopes WHERE name = 'Miscellaneous'")
        misc = cur.fetchone()
        
        if misc:
            misc_id = misc[0]
        else:
            misc_emb = compute_embedding("miscellaneous general")
            cur.execute(
                "INSERT INTO envelopes (name, topic_keywords, embedding, created_at) VALUES (?, ?, ?, ?)",
                ("Miscellaneous", json.dumps(["general"]), to_bytes(misc_emb), datetime.utcnow().isoformat())
            )
            misc_id = cur.lastrowid
        
        cur.execute("UPDATE cards SET envelope_id = ? WHERE envelope_id = ?", (misc_id, envelope_id))
        cur.execute("DELETE FROM envelopes WHERE id = ?", (envelope_id,))
        
        con.commit()
        con.close()
        
        return {"message": f"Envelope {envelope_id} deleted, cards moved to Miscellaneous"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting envelope: {str(e)}")

@app.post("/envelopes/cleanup")
async def api_cleanup_envelopes():
    """Cleanup duplicate Miscellaneous envelopes and merge them into one."""
    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        
        cur.execute("SELECT id, name FROM envelopes WHERE name = 'Miscellaneous'")
        misc_envs = cur.fetchall()
        
        if len(misc_envs) <= 1:
            con.close()
            return {"message": "No duplicate Miscellaneous envelopes found", "merged": 0}
        
        keep_id = misc_envs[0][0]
        merge_ids = [env[0] for env in misc_envs[1:]]
        
        for mid in merge_ids:
            cur.execute("UPDATE cards SET envelope_id = ? WHERE envelope_id = ?", (keep_id, mid))
            cur.execute("DELETE FROM envelopes WHERE id = ?", (mid,))
        
        con.commit()
        con.close()
        
        return {
            "message": f"Merged {len(merge_ids)} duplicate Miscellaneous envelope(s) into envelope {keep_id}",
            "merged": len(merge_ids),
            "kept_envelope_id": keep_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cleaning up envelopes: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)