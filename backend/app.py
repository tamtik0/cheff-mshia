"""
AI Recipe ChatBot with GROQ API (High limits, works with Georgian)
"""

from flask import Flask, request, render_template, session, redirect, url_for
from flask_cors import CORS
from dotenv import load_dotenv
import requests
import os
import json
import uuid
import re
import html
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

HISTORY_FL = os.path.join(DATA_DIR, 'chat_hist.json')
FAVORITES_FL = os.path.join(DATA_DIR, 'favorites.json')
USER_MEM_FL = os.path.join(DATA_DIR, 'user_memory.json')

TEMPLT_DIR = os.path.join(PROJECT_ROOT, 'templates')
if not os.path.isdir(TEMPLT_DIR):
    TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

app = Flask(__name__, template_folder=TEMPLT_DIR)

# Use environment variable for secret key (generate a new one for production!)
# To generate: import secrets; print(secrets.token_hex(32))
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

CORS(app)

# GROQ API - MUCH higher limits (3600/hour) and supports Georgian!
# Get API key from environment variable (NEVER hardcode in production!)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set! Create a .env file with GROQ_API_KEY=your_key")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Short factual guardrails for high-frequency Georgian dishes.
# These are injected into the system prompt and treated as canonical.


def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_json(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_history():
    data = load_json(HISTORY_FL)
    if isinstance(data, list):
        return data
    return data.get('chats', [])

def save_history(chats):
    save_json(HISTORY_FL, {'chats': chats})

def get_fav():
    data = load_json(FAVORITES_FL)
    if isinstance(data, list):
        return data
    return data.get('recipes', [])

def get_user_history(user_id):
    if not user_id:
        return []
    return [chat for chat in get_history() if chat.get('user_id') == user_id]

def get_user_fav(user_id):
    if not user_id:
        return []
    return [recipe for recipe in get_fav() if recipe.get('user_id') == user_id]

def save_fav(recipes):
    save_json(FAVORITES_FL, {'recipes': recipes})

def get_user_memory():
    memory = load_json(USER_MEM_FL)
    if not isinstance(memory, dict):
        memory = {}
    memory.setdefault('corrected_recipes', [])
    memory.setdefault('georgian_lexicon', [])
    memory.setdefault('georgian_notes', [])
    return memory

def save_user_memory(memory):
    save_json(USER_MEM_FL, memory)

def extract_urls(text):
    return re.findall(r'https?://[^\s<>"\]]+', text)

def get_rcp_url(url):
    try:
        response = requests.get(
            url,
            timeout=12,
            headers={"User-Agent": "Mozilla/5.0 (ChefMooRecipeBot)"}
        )
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text' not in content_type and 'html' not in content_type:
            return None

        raw_html = response.text
        # Remove non-content blocks first.
        cleaned = re.sub(r'<script.*?>.*?</script>', ' ', raw_html, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r'<style.*?>.*?</style>', ' ', cleaned, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r'<noscript.*?>.*?</noscript>', ' ', cleaned, flags=re.IGNORECASE | re.DOTALL)
        # Strip tags and normalize whitespace.
        text = re.sub(r'<[^>]+>', ' ', cleaned)
        text = html.unescape(text)
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            return None
        return text[:4000]
    except Exception:
        return None

def infer_recipe_subject(text):
    lowered = text.lower()
    patterns = [
        r'(?:recipe for|correct recipe for|for)\s+([a-z\u10A0-\u10FF][a-z\u10A0-\u10FF\s\-]{2,60})',
        r'(?:რეცეპტი|რეცეპტი\s+თვის)\s+([a-z\u10A0-\u10FF][a-z\u10A0-\u10FF\s\-]{2,60})'
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered, re.IGNORECASE)
        if match:
            return match.group(1).strip(" .,!?:;")
    return None

def correcting_stuff(text):
    lowered = text.lower()
    correction_markers = [
        'correct recipe', 'this is correct', 'this is wrong', 'wrong recipe',
        'should be', 'actually', 'remember this', 'save this recipe correction',
        'fix this recipe', 'არასწორია', 'სწორი რეცეპტი', 'შეასწორე', 'დაიმახსოვრე'
    ]
    return any(marker in lowered for marker in correction_markers)

def right_rcp_save(user_memory, user_msg):
    if not correcting_stuff(user_msg):
        return False

    subject = infer_recipe_subject(user_msg) or "general"
    entry = {
        'subject': subject,
        'correction': user_msg.strip(),
        'updated_at': datetime.now().isoformat()
    }

    corrected_recipes = user_memory.get('corrected_recipes', [])
    # Replace existing correction for the same subject, keep latest.
    corrected_recipes = [item for item in corrected_recipes if item.get('subject') != subject]
    corrected_recipes.insert(0, entry)
    user_memory['corrected_recipes'] = corrected_recipes[:50]
    return True

def get_corrections(user_memory, user_msg):
    corrected = user_memory.get('corrected_recipes', [])
    if not corrected:
        return []

    lowered_message = user_msg.lower()
    matched = []
    for item in corrected:
        subject = item.get('subject', '').lower()
        if subject != 'general' and subject and subject in lowered_message:
            matched.append(item)

    if not matched:
        # Fall back to latest general correction notes.
        matched = [item for item in corrected if item.get('subject') == 'general'][:2]

    return matched[:3]

def parse_georgian_learning_pairs(text):
    
    entries = []
    for raw_line in text.splitlines():
        line = raw_line.strip().strip('-*')
        if len(line) < 3:
            continue
        parts = re.split(r'\s*(?:=|:| - | – )\s*', line, maxsplit=1)
        if len(parts) != 2:
            continue
        source = parts[0].strip()
        meaning = parts[1].strip()
        if not source or not meaning:
            continue
        if not re.search(r'[\u10A0-\u10FF]', source):
            continue
        entries.append({'source': source[:80], 'meaning': meaning[:180]})
    return entries[:40]

def save_geo(user_memory, user_msg):
    lowered = user_msg.lower()
    explicit_markers = [
        'learn:', 'teach:', 'remember', 'save this georgian',
        'learn georgian', 'teaching you georgian', 'დაიმახსოვრე', 'ისწავლი'
    ]
    has_explicit_learning_intent = any(marker in lowered for marker in explicit_markers)
    parsed_pairs = parse_georgian_learning_pairs(user_msg)

    saved_count = 0

    if parsed_pairs and (has_explicit_learning_intent or len(parsed_pairs) >= 2):
        lexicon = user_memory.get('georgian_lexicon', [])
        lex_map = {item.get('source', '').lower(): item for item in lexicon}
        for item in parsed_pairs:
            lex_map[item['source'].lower()] = {
                'source': item['source'],
                'meaning': item['meaning'],
                'updated_at': datetime.now().isoformat()
            }
            saved_count += 1
        user_memory['georgian_lexicon'] = list(lex_map.values())[:300]

    # Save long grammar note blocks when user explicitly asks to remember.
    grammar_markers = ['grammar', 'rule', 'rules', 'წესი', 'გრამატიკა']
    if has_explicit_learning_intent and any(marker in lowered for marker in grammar_markers):
        notes = user_memory.get('georgian_notes', [])
        notes.insert(0, {
            'note': user_msg.strip()[:2500],
            'updated_at': datetime.now().isoformat()
        })
        user_memory['georgian_notes'] = notes[:40]
        saved_count += 1

    return saved_count

def get_relevant_georgian_learning(user_memory, user_msg):
    lexicon = user_memory.get('georgian_lexicon', [])
    notes = user_memory.get('georgian_notes', [])
    if not lexicon and not notes:
        return {'pairs': [], 'notes': []}

    text = user_msg.lower()
    pairs = []
    for item in lexicon:
        source = item.get('source', '').lower()
        meaning = item.get('meaning', '').lower()
        if source and source in text:
            pairs.append(item)
        elif meaning and len(meaning) > 2 and meaning in text:
            pairs.append(item)

    # Always include a few latest items so the bot can use what it learned.
    if not pairs:
        pairs = lexicon[:8]
    else:
        pairs = pairs[:12]

    return {
        'pairs': pairs,
        'notes': notes[:2]
    }

def extract_preferences(text):
    text_lower = text.lower()
    memory_updates = {}
    allergies = []
    
    if any(k in text_lower for k in ['თხილი', 'nuts', 'peanut']):
        allergies.append('nuts')
    if any(k in text_lower for k in ['გლუტენი', 'gluten']):
        allergies.append('gluten')
    if any(k in text_lower for k in ['რძე', 'milk', 'dairy']):
        allergies.append('dairy')
    if any(k in text_lower for k in ['კვერცხი', 'egg']):
        allergies.append('eggs')
    if any(k in text_lower for k in ['თევზი', 'fish']):
        allergies.append('seafood')
        
    if allergies:
        memory_updates['allergies'] = allergies
    
    diets = []
    if any(k in text_lower for k in ['მარხვა', 'fasting']):
        diets.append('orthodox_fasting')
    if any(k in text_lower for k in ['ვეგანი', 'vegan']):
        diets.append('vegan')
    if any(k in text_lower for k in ['ვეგეტარიანელი', 'vegetarian']):
        diets.append('vegetarian')
        
    if diets:
        memory_updates['diet'] = diets
    
    return memory_updates

def detect_message_language(text):
    text_lower = text.lower()

    # Respect explicit language requests first.
    if any(k in text_lower for k in ['in english', 'english please', 'respond in english', 'ინგლისურად']):
        return 'English'
    if any(k in text_lower for k in ['in georgian', 'respond in georgian', 'ქართულად']):
        return 'Georgian'

    # Decide by dominant script so Georgian dish names inside an English sentence
    # do not force the whole response into Georgian.
    eng_count = len(re.findall(r'[A-Za-z]', text))
    geo_count = len(re.findall(r'[\u10A0-\u10FF]', text))

    if eng_count >= geo_count:
        return 'English'
    return 'Georgian'

def get_requested_diet(text):
    text_lower = text.lower()
    if any(k in text_lower for k in ['მარხვა', 'fasting']):
        return 'orthodox_fasting'
    if any(k in text_lower for k in ['ვეგანი', 'vegan']):
        return 'vegan'
    if any(k in text_lower for k in ['ვეგეტარიანელი', 'vegetarian']):
        return 'vegetarian'
    return None



def build_prompt(user_memory, user_message, page_context=None):
    message_language = detect_message_language(user_message)
    requested_diet = get_requested_diet(user_message)
    relevant_corrections = get_corrections(user_memory, user_message)
    geo_learning = get_relevant_georgian_learning(user_memory, user_message)
    

    system = """You are Chef Mshia, a helpful recipe assistant. 
Respond ONLY in the same language as the user's current message.
Never switch languages unless the user explicitly asks you to.
If the user writes in English but mentions Georgian dish names (for example ქართული words),
still respond fully in English.
If the user asks for a translation, translate and continue in the requested language.

Default behavior:
- Give a normal/traditional version of the requested dish.
- Do NOT apply fasting,სამარხო, vegan,ვეგანური, vegetarian,ვეგეტარიანული, allergy-safe,არა ალერგიული, gluten-free, or dairy-free changes
  unless the user explicitly asks for those restrictions in the current message.
- If user asks for a Georgian dish, prioritize authentic Georgian culinary tradition and canonical ingredients/technique.
- Never silently swap to a different cuisine dish; if unsure, say what is uncertain and ask one short clarification question.
- Never invent new regional dish names (example of forbidden style: "Kartli Sulguni" unless user gave a trusted source).

Format recipes like this:
**Recipe Name:** [Name 🍳] 
**Time:** [prep time 🕒]
**Difficulty:** [Easy/Medium/Hard ✅]

**Ingredients:**
* ingredient 1
* ingredient 2

**Instructions:**
1. Step one
2. Step two
3. Step three"""

    system += f"\n\nCurrent user message language: {message_language}"
    if requested_diet:
        system += f"\nDiet requested in current message: {requested_diet}"
    else:
        system += "\nNo diet restriction requested in current message."

    if relevant_corrections:
        system += "\n\nPreviously saved recipe corrections from user (treat as preferred truth):"
        for item in relevant_corrections:
            system += f"\n- Subject: {item.get('subject', 'general')} | Note: {item.get('correction', '')}"

    if geo_learning.get('pairs'):
        system += "\n\nUser-taught Georgian vocabulary/preferences (use when helpful):"
        for item in geo_learning['pairs']:
            system += f"\n- {item.get('source', '')} => {item.get('meaning', '')}"

    if geo_learning.get('notes'):
        system += "\n\nUser-taught Georgian grammar notes:"
        for note in geo_learning['notes']:
            system += f"\n- {note.get('note', '')[:600]}"

    
    if page_context:
        system += (
            "\n\nRecipe page context provided by user link (use this as source material when relevant):\n"
            f"{page_context}"
        )
    
    return system

def extract_ai_message(result):
    """
    Safely extract model text from GROQ/OpenAI-like responses.
    Returns (message, error_message). Exactly one will be non-empty.
    """
    if not isinstance(result, dict):
        return "", "Unexpected API response format."

    if result.get('error'):
        err = result.get('error')
        if isinstance(err, dict):
            return "", err.get('message') or str(err)
        return "", str(err)

    choices = result.get('choices')
    if not isinstance(choices, list) or not choices:
        return "", "No reply choices were returned by the AI service."

    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get('message', {}) if isinstance(first, dict) else {}
    content = message.get('content') if isinstance(message, dict) else None

    if isinstance(content, str) and content.strip():
        return content, ""

    # Fallback for providers that sometimes use text in different field.
    alt_text = first.get('text') if isinstance(first, dict) else None
    if isinstance(alt_text, str) and alt_text.strip():
        return alt_text, ""

    return "", "The AI service returned an empty response."

current_sessions = {}

def get_current_chat(session_id):
    if session_id not in current_sessions:
        current_sessions[session_id] = {
            'id': str(uuid.uuid4()),
            'title': 'New Chat',
            'messages': [],
            'created_at': datetime.now().isoformat()
        }
    return current_sessions[session_id]

def format_response(text):
    # Convert markdown to HTML
    text = text.replace('**', '<strong>').replace('**', '</strong>')
    text = text.replace('\n\n', '<br><br>')
    text = text.replace('\n', '<br>')
    return text

@app.route('/')
def serve_index():
    if 'username' not in session:
        return render_template('index.html', username=None)
    
    session_id = session.get('user_id', str(uuid.uuid4()))
    if 'user_id' not in session:
        session['user_id'] = session_id
    
    current_chat = get_current_chat(session_id)
    
    # Format messages for display
    formatted_messages = []
    for msg in current_chat['messages']:
        formatted_msg = msg.copy()
        if msg['role'] == 'assistant':
            formatted_msg['content'] = format_response(msg['content'])
        formatted_messages.append(formatted_msg)
    
    return render_template('index.html', 
                         username=session['username'],
                         messages=formatted_messages,
                         error=session.pop('error', None))

@app.route('/set_name', methods=['POST'])
def set_name():
    username = request.form.get('username', '').strip()
    if username:
        session.clear()
        session['username'] = username
        session['user_id'] = str(uuid.uuid4())
        current_sessions[session['user_id']] = {
            'id': str(uuid.uuid4()),
            'title': 'New Chat',
            'messages': [],
            'created_at': datetime.now().isoformat(),
            'user_id': session['user_id']
        }
    return redirect(url_for('serve_index'))

@app.route('/chat', methods=['POST'])
def chat():
    if 'username' not in session:
        return redirect(url_for('serve_index'))
    
    user_msg = request.form.get('message', '').strip()
    if not user_msg:
        return redirect(url_for('serve_index'))
    
    session_id = session.get('user_id')
    current_chat = get_current_chat(session_id)
    
    # Load memory and save corrections when user provides them.
    user_memory = get_user_memory()
    correction_saved = right_rcp_save(user_memory, user_msg)
    learning_saved_count = save_geo(user_memory, user_msg)
    if correction_saved or learning_saved_count:
        save_user_memory(user_memory)

    # Pull recipe page context when user includes a URL.
    page_context = None
    urls = extract_urls(user_msg)
    for url in urls[:2]:
        fetched = get_rcp_url(url)
        if fetched:
            page_context = f"Source URL: {url}\n{fetched}"
            break
    
    # Add user message
    current_chat['messages'].append({
        'role': 'user',
        'content': user_msg
    })
    
    # Build system prompt
    system_prompt = build_prompt(user_memory, user_msg, page_context=page_context)
    
    try:
        # Include recent conversation so assistant keeps context/memory in this chat.
        # We keep only the latest messages to avoid oversized requests.
        conversation_context = current_chat['messages'][-12:]
        api_messages = [{"role": "system", "content": system_prompt}] + conversation_context

        # Call GROQ API (high limits!)
        response = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": api_messages,
                "max_tokens": 2000,
                "temperature": 0.7
            },
            timeout=30
        )
        
        if response.status_code >= 400:
            try:
                error_result = response.json()
                _, api_error = extract_ai_message(error_result)
                if not api_error:
                    api_error = str(error_result)
            except Exception:
                api_error = response.text[:300] or f"HTTP {response.status_code}"
            raise RuntimeError(f"AI API error ({response.status_code}): {api_error}")

        result = response.json()
        ai_response, parse_error = extract_ai_message(result)
        if parse_error:
            raise RuntimeError(parse_error)
        if learning_saved_count:
            ai_response = (
                f"Saved {learning_saved_count} Georgian learning note(s) to memory.\n\n"
                f"{ai_response}"
            )
        
        # Add AI response
        current_chat['messages'].append({
            'role': 'assistant',
            'content': ai_response
        })
        
    except Exception as e:
        session['error'] = f"Error: {str(e)}"
    
    return redirect(url_for('serve_index'))

@app.route('/new_chat', methods=['POST'])
def new_chat():
    if 'user_id' in session:
        session_id = session['user_id']
        current_sessions[session_id] = {
            'id': str(uuid.uuid4()),
            'title': 'New Chat',
            'messages': [],
            'created_at': datetime.now().isoformat()
        }
    return redirect(url_for('serve_index'))

@app.route('/save_chat', methods=['POST'])
def save_current_chat():
    if 'user_id' not in session:
        return redirect(url_for('serve_index'))
    
    session_id = session['user_id']
    current_chat = current_sessions.get(session_id)
    
    if not current_chat or not current_chat['messages']:
        return redirect(url_for('serve_index'))
    
    chats = get_history()
    existing = next((c for c in chats if c['id'] == current_chat['id']), None)
    
    chat_data = {
        'id': current_chat['id'],
        'user_id': session_id,
        'title': current_chat['title'],
        'messages': current_chat['messages'],
        'created_at': current_chat['created_at'],
        'updated_at': datetime.now().isoformat()
    }
    
    if existing:
        idx = chats.index(existing)
        chats[idx] = chat_data
    else:
        chats.insert(0, chat_data)
    
    chats = chats[:50]
    save_history(chats)
    return redirect(url_for('serve_index'))

@app.route('/history')
def show_history():
    if 'username' not in session:
        return redirect(url_for('serve_index'))
    chats = get_user_history(session.get('user_id'))
    return render_template('history.html', username=session['username'], chats=chats)

@app.route('/favorites')
def show_fav():
    if 'username' not in session:
        return redirect(url_for('serve_index'))
    fav = get_user_fav(session.get('user_id'))
    return render_template('favorites.html', username=session['username'], favorites=fav)

@app.route('/load_chat/<chat_id>')
def load_chat(chat_id):
    if 'user_id' not in session:
        return redirect(url_for('serve_index'))
    
    chats = get_user_history(session['user_id'])
    chat = next((c for c in chats if c['id'] == chat_id), None)
    
    if chat:
        session_id = session['user_id']
        current_sessions[session_id] = {
            'id': chat['id'],
            'title': chat['title'],
            'messages': chat['messages'],
            'created_at': chat['created_at'],
            'user_id': session_id
        }
    
    return redirect(url_for('serve_index'))

@app.route('/delete_chat/<chat_id>', methods=['POST'])
def del_chat(chat_id):
    if 'user_id' not in session:
        return redirect(url_for('serve_index'))
    user_id = session['user_id']
    chats = get_history()
    chats = [c for c in chats if not (c.get('id') == chat_id and c.get('user_id') == user_id)]
    save_history(chats)
    return redirect(url_for('show_history'))

@app.route('/save_favorite', methods=['POST'])
def save_favorite():
    content = request.form.get('content', '')
    if not content or 'user_id' not in session:
        return redirect(url_for('serve_index'))
    
    # Extract title
    import re
    title_match = re.search(r'Recipe Name:(.*?)(<|$)', content, re.IGNORECASE)
    if title_match:
        title = title_match.group(1).strip()
    else:
        title = content.split('\n')[0][:50]
    
    clean_content = re.sub(r'<[^>]+>', '', content)
    
    recipes = get_fav()
    recipe = {
        'id': str(uuid.uuid4()),
        'user_id': session['user_id'],
        'title': title or 'Recipe',
        'content': clean_content,
        'created_at': datetime.now().isoformat()
    }
    
    recipes.insert(0, recipe)
    save_fav(recipes)
    return redirect(url_for('serve_index'))

@app.route('/delete_favorite/<fav_id>', methods=['POST'])
def delete_favorite(fav_id):
    if 'user_id' not in session:
        return redirect(url_for('serve_index'))
    user_id = session['user_id']
    recipes = get_fav()
    recipes = [r for r in recipes if not (r.get('id') == fav_id and r.get('user_id') == user_id)]
    save_fav(recipes)
    return redirect(url_for('show_favorites'))

@app.route('/clear_memory', methods=['POST'])
def clear_memory():
    save_user_memory({})
    return redirect(url_for('serve_index'))

if __name__ == '__main__':
    print("=" * 50)
    print("Chef AI with GROQ API")
    print("Limits: 3,600 requests/hour")
    print("=" * 50)
    # For local development only. In production, use gunicorn.
    app.run(debug=True, host='127.0.0.1', port=5000)
