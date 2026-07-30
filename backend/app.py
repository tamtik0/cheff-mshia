"""
AI Recipe ChatBot with GROQ API (High limits, works with Georgian)
"""
# importing main engine 
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flask import Flask, request, render_template, session, redirect, url_for
from flask_cors import CORS #talk to other sites
from dotenv import load_dotenv # reads .env
from markupsafe import escape, Markup
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import requests #getting data from other sites
import os #int wth cmptr os fr fndign files
import json #rd/wrt jsn files
import uuid#gnrt unique ids fr usr/ct/fav
import re #find patrn in txt
import html 
from datetime import datetime  #fr timestamps
import socket
import ipaddress
from urllib.parse import urlparse
import tempfile
from db import supabase
from datetime import date

#---------setup-----------
# Load environment variables from .env file
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #fldr cont app.py
PROJECT_ROOT = os.path.dirname(BASE_DIR) # prnt pf base_dir
DATA_DIR = os.path.join(BASE_DIR, 'data') #backend/dt
os.makedirs(DATA_DIR, exist_ok=True) #ens dt dir exist without crashinf if alrd prst
# create path for files
HISTORY_FL = os.path.join(DATA_DIR, 'chat_hist.json')
FAVORITES_FL = os.path.join(DATA_DIR, 'favorites.json')
USER_MEM_FL = os.path.join(DATA_DIR, 'user_memory.json')
#chs templ dir
TEMPLT_DIR = os.path.join(PROJECT_ROOT, 'templates') # frst try
if not os.path.isdir(TEMPLT_DIR):
    TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
# crt flsk app points to templt_dir for html temp
app = Flask(__name__, template_folder=TEMPLT_DIR)# Initialize the Flask app


# Security key for sessions
app.secret_key = os.environ.get('SECRET_KEY')
if not app.secret_key:
    raise ValueError("SECRET_KEY environment variable not set! Set it in Render's Environmnet tab.")

app.config['SESSION_COOKIE_SECURE'] = True      # cookie only sent over HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True     # already default, but explicit is safer
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'    # explicit CSRF baseline, don't rely on browser defaults
# Restrict cross-origin requests to just this app's own domain
CORS(app, origins=["https://cheff-mshia.onrender.com"])

# Rate limiting: protects the shared Groq key and prevents one user
# from exhausting the app for everyone else
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per hour"],
    storage_uri="memory://",
)

# get api key from environmet variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
#stop apps and shows errors if not found
if not GROQ_API_KEY: 
    raise ValueError("GROQ_API_KEY environment variable not set! Create a .env file with GROQ_API_KEY=your_key")
#link to where it gets the key
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


#---------small functions helping---------

def load_json(filepath):
    """Read JSON data from disk and return an empty dict if the file is missing."""
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}
#saves data into json fl
def save_json(filepath, data):
    """Persist JSON atomically to avoid corruption from concurrent writes."""
    dir_name = os.path.dirname(filepath)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, filepath)  # atomic on POSIX
    except Exception:
        os.unlink(tmp_path)
        raise

def get_history():
    """Load chat history and normalize legacy/modern file formats to a chat list."""
    data = load_json(HISTORY_FL)
    if isinstance(data, list): #i simple list return same
        return data
    return data.get('chats', [])#if dictionary look for chat,is notfound return empty

def save_history(chats):
    """Write all chats to storage using a keyed object format."""
    save_json(HISTORY_FL, {'chats': chats})

def get_fav():
    """Load favorite recipes and normalize legacy/modern file formats to a recipe list."""
    data = load_json(FAVORITES_FL)
    if isinstance(data, list):
        return data
    return data.get('recipes', [])

def get_user_history(user_id):
    if not user_id:
        return []
    result = supabase.table('chats').select('*').eq('user_id', user_id).order('created_at', desc=True).execute()
    return result.data

def create_chat(user_id):
    result = supabase.table('chats').insert({'user_id': user_id, 'title': 'New Chat'}).execute()
    return result.data[0]

def signup_user(email, password):
    response = supabase.auth.sign_up({"email": email, "password": password})
    # Safely get the user id
    return response.user.id if response and response.user else None

def login_user(email, password):
    response = supabase.auth.sign_in_with_password({"email": email, "password": password})
    return response.user.id if response and response.user else None


def add_message(chat_id, role, content):
    result = supabase.table('messages').insert({
        'chat_id': chat_id,
        'role': role,
        'content': content
    }).execute()
    return result.data[0]

def get_chat_messages(chat_id):
    """Fetch all messages for a given chat, oldest first."""
    if not chat_id:
        return []
    result = supabase.table('messages') \
        .select('*') \
        .eq('chat_id', chat_id) \
        .order('created_at') \
        .execute()
    return result.data


def get_user_fav(user_id):
    if not user_id:
        return []
    result = supabase.table('favorites').select('*').eq('user_id', user_id).order('created_at', desc=True).execute()
    return result.data

def add_favorite(user_id, title, content):
    result = supabase.table('favorites').insert({
        'user_id': user_id,
        'title': title,
        'content': content
    }).execute()
    return result.data[0]

def get_user_memory(user_id):
    result = supabase.table('user_memory').select('*').eq('user_id', user_id).execute()
    if result.data:
        return result.data[0]
    return {'user_id': user_id, 'corrected_recipes': [], 'georgian_lexicon': [], 'georgian_notes': []}

def save_user_memory(user_id, memory):
    memory['user_id'] = user_id
    supabase.table('user_memory').upsert(memory).execute()

def extract_urls(text):
   # Scans the text and returns a list of all web addresses found
    return re.findall(r'https?://[^\s<>"\]]+', text)

def is_safe_url(url):
    """Reject URLs pointing at internal/private/loopback addresses to prevent SSRF."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            return False
        hostname = parsed.hostname
        if not hostname:
            return False
        for family , _, _, _, sockaddr in socket.getaddrinfo(hostname, None):
            ip = ipaddress.ip_address(sockaddr[0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return False
        return True
    except Exception:
        return False

    

def get_rcp_url(url):
    """
    Fetch a recipe page and convert HTML into compact plain text context.
    Returns None when fetch/parsing fails or content is not text-like.
    """
    try:
        if not is_safe_url(url):
            return None
        response = requests.get(
            url,
            timeout=12,
            allow_redirects=False,
            headers={"User-Agent": "Mozilla/5.0 (ChefMshiaRecipeBot)"}
        )
        if response.status_code in (301, 302, 303, 307, 308):
            return None
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text' not in content_type and 'html' not in content_type:
            return None
        content_length = response.headers.get('Content-Length')
        if content_length and int(content_length) > 2_000_000:
            return None
        if len(response.content) > 2_000_000:
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
    #guessing which recipe is it 
    lowered = text.lower()
    patterns = [
        r'(?:recipe for|correct recipe for|for)\s+([a-z\u10A0-\u10FF][a-z\u10A0-\u10FF\s\-]{2,60})',
        r'(?:რეცეპტი|რეცეპტი\s+თვის)\s+([a-z\u10A0-\u10FF][a-z\u10A0-\u10FF\s\-]{2,60})'
    ]
    for pattern in patterns:#fidning patters
        match = re.search(pattern, lowered, re.IGNORECASE)
        if match: #if found sendit 
            return match.group(1).strip(" .,!?:;")
    return None

def correcting_stuff(text):
    #to correct wrong stuff
    lowered = text.lower()
    #to know they are wrong and correct by this words
    correction_markers = [
        'correct recipe', 'this is correct', 'this is wrong', 'wrong recipe',
        'should be', 'actually', 'remember this', 'save this recipe correction',
        'fix this recipe', 'არასწორია', 'სწორი რეცეპტი', 'შეასწორე', 'დაიმახსოვრე'
    ]
    return any(marker in lowered for marker in correction_markers)

def right_rcp_save(user_memory, user_msg):
    #saving right recip after correction
    if not correcting_stuff(user_msg):
        return False

    subject = infer_recipe_subject(user_msg) or "general"
    entry = {
        'subject': subject,
        'correction': user_msg.strip(),
        'updated_at': datetime.now().isoformat()
    }
#saving corrected recipes and giving it to us 
    corrected_recipes = user_memory.get('corrected_recipes', [])
    # Replace existing correction for the same subject, keep latest.
    corrected_recipes = [item for item in corrected_recipes if item.get('subject') != subject]
    corrected_recipes.insert(0, entry)
    user_memory['corrected_recipes'] = corrected_recipes[:50]
    return True

def get_corrections(user_memory, user_msg):
   #returning correct stuff
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
    """Parse Georgian learning pairs like 'word = meaning' from user input."""
    
    entries = []
    for raw_line in text.splitlines():
        line = raw_line.strip().strip('-*') #removing this kinda stuff
        if len(line) < 3:
            continue
        #splitting pairs
        parts = re.split(r'\s*(?:=|:| - | – )\s*', line, maxsplit=1)
        if len(parts) != 2:
            continue
        source = parts[0].strip()
        meaning = parts[1].strip()
        if not source or not meaning:
            continue
        #checking for georgian
        if not re.search(r'[\u10A0-\u10FF]', source):
            continue
        #seving results
        entries.append({'source': source[:80], 'meaning': meaning[:180]})
    return entries[:40]

def save_geo(user_memory, user_msg):
    #save user tought things
    lowered = user_msg.lower()
    explicit_markers = [ #using this to make it save and remember
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
    #pickes usefull saved geo stuff
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
    #for diet allergy stuff
    text_lower = text.lower()
    memory_updates = {}
    allergies = []
    #if any of it saving it 
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
        #updates memory if any of them
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
    #detect which language is it mostly used in mmsgs
    text_lower = text.lower()

    # Respect explicit language requests first.
    if any(k in text_lower for k in ['in english', 'english please', 'respond in english', 'ინგლისურად']):
        return 'English'
    if any(k in text_lower for k in ['in georgian', 'respond in georgian', 'ქართულად']):
        return 'Georgian'

    ##choosing which language to use for response
    eng_count = len(re.findall(r'[A-Za-z]', text)) #counting the length
    geo_count = len(re.findall(r'[\u10A0-\u10FF]', text))
#uses the longer language to response
    if eng_count >= geo_count:
        return 'English'
    return 'Georgian'

def get_requested_diet(text):
    #checks if diets stuff is asked 
    text_lower = text.lower()
    if any(k in text_lower for k in ['მარხვა', 'fasting']):
        return 'orthodox_fasting'
    if any(k in text_lower for k in ['ვეგანი', 'vegan']):
        return 'vegan'
    if any(k in text_lower for k in ['ვეგეტარიანელი', 'vegetarian']):
        return 'vegetarian'
    return None



def build_prompt(user_memory, user_message, page_context=None):
    """
    Build the full system prompt by combining policy, memory, and page context.
    This keeps the model aligned with language/diet/memory constraints.
    """
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
#Safely extract model text from GROQ/OpenAI-like responses.
    # Check if the response is a valid dictionary; if not, return an error
    
    if not isinstance(result, dict):
        return "", "Unexpected API response format."
# If the AI service reported an error (like an invalid API key), extract and return that error
    if result.get('error'):
        err = result.get('error')
        if isinstance(err, dict):
            return "", err.get('message') or str(err)
        return "", str(err)
# Drill down into the 'choices' list where the actual message lives
    choices = result.get('choices')
    if not isinstance(choices, list) or not choices:
        return "", "No reply choices were returned by the AI service."
# Navigate the nested layers to find the 'content' string (the AI's actual answer)
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get('message', {}) if isinstance(first, dict) else {}
    content = message.get('content') if isinstance(message, dict) else None
# If content is found, return the message; otherwise, check an alternative field ('text')
    if isinstance(content, str) and content.strip():
        return content, ""

    # Fallback for providers that sometimes use text in different field.
    alt_text = first.get('text') if isinstance(first, dict) else None
    if isinstance(alt_text, str) and alt_text.strip():
        return alt_text, ""

    return "", "The AI service returned an empty response."

current_sessions = {}
SESSION_TTL_SECONDS = 3600  # 1 hour of inactivity

def prune_old_sessions():
    now = datetime.now()
    stale = []
    for sid, chat in current_sessions.items():
        try:
            created = datetime.fromisoformat(chat['created_at'])
            if (now - created).total_seconds() > SESSION_TTL_SECONDS:
                stale.append(sid)
        except Exception:
            stale.append(sid)
    for sid in stale:
        current_sessions.pop(sid, None)

def get_current_chat(session_id):
    #gets in current chat if dont exist creates it 
    prune_old_sessions()
    if session_id not in current_sessions:
        current_sessions[session_id] = {
            'id': str(uuid.uuid4()),
            'title': 'New Chat',
            'messages': [],
            'created_at': datetime.now().isoformat()
        }
    return current_sessions[session_id]

def format_response(text):
    """Escape user/AI content first, then apply safe minimal formatting."""
    # Convert markdown to HTML
    escaped = str(escape(text))
    escaped = escaped.replace('**', '<strong>', 1).replace('**', '</strong>', 1)
    escaped = escaped.replace('\n\n', '<br><br>').replace('\n', '<br>')
    return Markup(escaped)
#--------web routes----------
@app.route('/robots.txt')
def robots():
    return "User-agent: *\nAllow: /", 200, {'Content-Type': 'text/plain'}

    
@app.route('/')
def serve_index():
    if 'user_id' not in session:
        return render_template('index.html', username=None, error=session.pop('error', None))
    session_id = session['user_id']
    #getting active chat msgs from this user 
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

@app.route('/signup', methods=['POST'])
def signup():
    email = request.form.get('email', '').strip()
    password = request.form.get('password', '')
    if not email or len(password) < 8:
        session['error'] = "Password must be at least 8 characters."
        return redirect(url_for('serve_index'))
    try:
        signup_user(email, password)
        session['error'] = "Account created! Check your email to confirm it, then log in below."
    except Exception as e:
        session['error'] = f"Signup failed: {str(e)}"
    return redirect(url_for('serve_index'))

@app.route('/login', methods=['POST'])
def login():
    email = request.form.get('email', '').strip()
    password = request.form.get('password', '')
    try:
        user_id = login_user(email, password)
        session.clear()
        session['user_id'] = user_id
        session['username'] = email.split('@')[0]
    except Exception:
        session['error'] = "Invalid email or password, or your account isn't confirmed yet."
    return redirect(url_for('serve_index'))

@app.route('/terms')
def terms():
    return render_template('terms.html', today=date.today().strftime("%B %d, %Y"), contact_email="termakozashvilitamta@gmail.com")

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return redirect(url_for('serve_index'))

@app.route('/auth/google')
def auth_google():
    result = supabase.auth.sign_in_with_oauth({
        "provider": "google",
        "options": {
            "redirect_to": request.host_url.rstrip('/') + "/auth/callback",
            "query_params": {"prompt": "select_account"}
        }
    })
    return redirect(result.url)


@app.route('/auth/callback')
def auth_callback():
    code = request.args.get('code')
    if not code:
        session['error'] = "Google sign-in was cancelled or failed."
        return redirect(url_for('serve_index'))
    try:
        # exchange_code_for_session requires code_verifier and redirect_to in some client implementations
        result = supabase.auth.exchange_code_for_session({
            "auth_code": code,
            "redirect_to": "https://cheff-mshia.onrender.com/auth/callback",
            "code_verifier": os.environ.get("SUPABASE_CODE_VERIFIER", "")
        })
        session.clear()
        session['user_id'] = result.user.id
        session['username'] = result.user.email.split('@')[0]
    except Exception as e:
        session['error'] = f"Google sign-in failed: {str(e)}"
    return redirect(url_for('serve_index'))

@app.route('/set_name', methods=['POST'])
def set_name():
    #saves username, creat4es new id , starts new chat
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

@app.route('/forgot_password', methods=['POST'])
def forgot_password():
    email = request.form.get('email', '').strip()
    try:
        supabase.auth.reset_password_email(email, {"redirect_to": "https://cheff-mshia.onrender.com/reset_password"})
        session['error'] = "If that email exists, a reset link has been sent."
    except Exception:
        session['error'] = "If that email exists, a reset link has been sent."
    return redirect(url_for('serve_index'))

@app.errorhandler(405)
def method_not_allowed(e):
    return redirect(url_for('serve_index'))

@app.route('/chat', methods=['POST'])
@limiter.limit("10 per minute; 100 per day")
def chat():
    #reades user memory
    if 'username' not in session:
        return redirect(url_for('serve_index'))
    
    user_msg = request.form.get('message', '').strip()[:3000]
    if not user_msg:
        return redirect(url_for('serve_index'))
    
    session_id = session.get('user_id')
    current_chat = get_current_chat(session_id)
    
    # Load memory and save corrections when user provides them.
    user_memory = get_user_memory(session_id)
    correction_saved = right_rcp_save(user_memory, user_msg)
    learning_saved_count = save_geo(user_memory, user_msg)
    if correction_saved or learning_saved_count:
        save_user_memory(session_id, user_memory)

    # Pull recipe page context when user includes a URL.
    page_context = None
    urls = extract_urls(user_msg)
    for url in urls[:2]:
        fetched = get_rcp_url(url)
        if fetched:
            page_context = f"Source URL: {url}\n{fetched}"
            break
    
    # Persist current user turn in the active in-memory conversation.
    current_chat['messages'].append({
        'role': 'user',
        'content': user_msg
    })
    
    # Build a prompt that includes behavior rules and relevant remembered context.
    system_prompt = build_prompt(user_memory, user_msg, page_context=page_context)
    
    try:
        # Send recent turns only, balancing context quality against token/request size.
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
        
        # Save assistant reply in active conversation for UI display and future context.
        current_chat['messages'].append({
            'role': 'assistant',
            'content': ai_response
        })
        
    except Exception as e:
        session['error'] = f"Error: {str(e)}"
    
    return redirect(url_for('serve_index'))

@app.route('/new_chat', methods=['POST'])
def new_chat():
    """Start a fresh in-memory chat for the current authenticated user."""
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
    # If this in-memory chat was never persisted yet, create the real row now
    existing = supabase.table('chats').select('id').eq('id', current_chat['id']).execute()
    if not existing.data:
        supabase.table('chats').insert({
            'id': current_chat['id'],
            'user_id': session_id,
            'title': current_chat['title']
        }).execute()
    return redirect(url_for('serve_index'))

@app.route('/history')
def show_history():
    """Render the history page with saved chats for the current user."""
    if 'username' not in session:
        return redirect(url_for('serve_index'))
    chats = get_user_history(session.get('user_id'))
    return render_template('history.html', username=session['username'], chats=chats)

@app.route('/favorites')
def show_fav():
    """Render the favorites page with saved recipes for the current user."""
    if 'username' not in session:
        return redirect(url_for('serve_index'))
    fav = get_user_fav(session.get('user_id'))
    return render_template('favorites.html', username=session['username'], favorites=fav)

@app.route('/load_chat/<chat_id>')
def load_chat(chat_id):
    """Load one saved chat into active in-memory session state."""
    if 'user_id' not in session:
        return redirect(url_for('serve_index'))
    
    chats = get_user_history(session['user_id'])
    # str(...) handles UUID vs string mismatch from Supabase
    chat = next((c for c in chats if str(c.get('id')) == chat_id), None)

    if chat:
        session_id = session['user_id']
        # Messages live in a separate Supabase table — fetch them separately
        current_sessions[session_id] = {
            'id': chat['id'],
            'title': chat.get('title', 'New Chat'),
            'messages': get_chat_messages(chat['id']),
            'created_at': chat.get('created_at'),
            'user_id': session_id
        }
    
    return redirect(url_for('serve_index'))

@app.route('/delete_chat/<chat_id>', methods=['POST'])
def del_chat(chat_id):
    """Delete one saved chat that belongs to the current user."""
    if 'user_id' not in session:
        return redirect(url_for('serve_index'))
    user_id = session['user_id']
    # Delete messages first (foreign key dependency), then the chat itself
    supabase.table('messages').delete().eq('chat_id', chat_id).execute()
    supabase.table('chats').delete().eq('id', chat_id).eq('user_id', user_id).execute()
    return redirect(url_for('show_history'))

@app.route('/save_favorite', methods=['POST'])
def save_favorite():
    """Save rendered recipe content as a favorite record for the current user."""
    content = request.form.get('content', '')
    if not content or 'user_id' not in session:
        return redirect(url_for('serve_index'))
    
    # Extract title from formatted recipe content
    title_match = re.search(r'Recipe Name:(.*?)(<|$)', content, re.IGNORECASE)
    if title_match:
        title = title_match.group(1).strip()
    else:
        title = content.split('\n')[0][:50]
    
    clean_content = re.sub(r'<[^>]+>', '', content)
    add_favorite(session['user_id'], title or 'Recipe', clean_content)
    return redirect(url_for('serve_index'))

@app.route('/delete_favorite/<fav_id>', methods=['POST'])
def delete_favorite(fav_id):
    """Delete one favorite recipe entry that belongs to the current user."""
    if 'user_id' not in session:
        return redirect(url_for('serve_index'))
    supabase.table('favorites').delete() \
        .eq('id', fav_id) \
        .eq('user_id', session['user_id']) \
        .execute()
    return redirect(url_for('show_fav'))

@app.route('/clear_memory', methods=['POST'])
def clear_memory():
    """Reset persisted user memory (corrections, vocabulary, grammar notes)."""
    if 'user_id' not in session:
        return redirect(url_for('serve_index'))
    save_user_memory(session['user_id'], {
        'corrected_recipes': [],
        'georgian_lexicon': [],
        'georgian_notes': []
    })
    return redirect(url_for('serve_index'))

if __name__ == '__main__':
    print("=" * 50)
    print("Chef AI with GROQ API")
    print("Limits: 3,600 requests/hour")
    print("=" * 50)
    # For local development only. In production, use gunicorn.
    app.run(debug=True, host='127.0.0.1', port=5000)
