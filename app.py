import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
import pickle

# Page configuration
st.set_page_config(
    page_title="NEXUS - Series Recommendation",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .stApp {
        background: #0a0e27;
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(0, 255, 163, 0.05) 0%, transparent 40%),
            radial-gradient(circle at 90% 80%, rgba(0, 255, 163, 0.03) 0%, transparent 40%),
            linear-gradient(180deg, #0a0e27 0%, #050817 100%);
    }

    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Logo and brand */
    .brand-logo {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00ffa3 0%, #00d4ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        letter-spacing: -1px;
        margin-bottom: 0.5rem;
    }

    .brand-tagline {
        text-align: center;
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.95rem;
        font-weight: 400;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 3rem;
    }

    /* Creative Welcome Header */
    .welcome-container {
        padding: 1rem 0 2rem 0;
        animation: fadeInUp 0.8s ease-out;
    }

    .welcome-title {
        font-size: 3.5rem;
        font-weight: 900;
        margin: 0;
        line-height: 1.2;
        background: linear-gradient(90deg, #ffffff 0%, #00ffa3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 40px rgba(0, 255, 163, 0.2);
        letter-spacing: -1px;
    }

    .welcome-subtitle {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.6);
        font-weight: 400;
        margin-top: 0.5rem;
        letter-spacing: 0.5px;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Glass morphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 2.5rem;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        border: 1px solid rgba(0, 255, 163, 0.3);
        box-shadow: 
            0 12px 48px rgba(0, 0, 0, 0.4),
            0 0 40px rgba(0, 255, 163, 0.1);
    }

    /* Input fields */
    .stTextInput > label {
        color: rgba(255, 255, 255, 0.8) !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        margin-bottom: 0.5rem !important;
    }

    .stTextInput input {
        background: #ffffff !important; 
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 0.875rem 1rem !important;
        color: #000000 !important; 
        font-size: 0.95rem !important;
        transition: all 0.3s ease !important;
    }

    .stTextInput input:focus {
        background: #ffffff !important;
        border: 1px solid #00ffa3 !important;
        box-shadow: 0 0 0 3px rgba(0, 255, 163, 0.1) !important;
    }

    .stTextInput input::placeholder {
        color: #666666 !important;
    }

    /* Modern buttons */
    .stButton button {
        background: linear-gradient(135deg, #00ffa3 0%, #00d4ff 100%) !important;
        color: #0a0e27 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.875rem 2rem !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 16px rgba(0, 255, 163, 0.3) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }

    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 24px rgba(0, 255, 163, 0.5) !important;
    }

    /* Tabs styling */
    .stTabs {
        background: transparent;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: rgba(255, 255, 255, 0.03);
        padding: 0.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        color: #ffffff;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(0, 255, 163, 0.1);
        color: #00ffa3;
    }

    /* Headers */
    h1 {
        color: #ffffff;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }

    h2 {
        color: #ffffff !important; 
        font-weight: 600;
        font-size: 1.75rem;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
    }

    h3 {
        color: rgba(255, 255, 255, 0.9);
        font-weight: 600;
        font-size: 1.25rem;
        margin-bottom: 1.5rem;
    }

    /* Series card */
    .series-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 1.75rem;
        margin: 1rem 0;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        display: flex;
        gap: 1.5rem;
        height: 100%;
    }

    .series-poster {
        flex-shrink: 0;
        width: 120px;
        height: 180px;
        border-radius: 12px;
        overflow: hidden;
        background: rgba(255, 255, 255, 0.05);
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .series-poster img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }

    .series-content {
        flex: 1;
        display: flex;
        flex-direction: column;
    }

    .series-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #00ffa3 0%, #00d4ff 100%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .series-card:hover {
        transform: translateY(-8px);
        border: 1px solid rgba(0, 255, 163, 0.3);
        box-shadow: 
            0 20px 60px rgba(0, 0, 0, 0.4),
            0 0 40px rgba(0, 255, 163, 0.15);
    }

    .series-card:hover::before {
        opacity: 1;
    }

    .series-title {
        color: #ffffff;
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
        letter-spacing: -0.3px;
    }

    .series-meta {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin: 1rem 0;
        flex-wrap: wrap;
    }

    .series-genres {
         margin-bottom: 0.75rem;
         flex-wrap: wrap;
    }

    .series-rating {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(255, 193, 7, 0.15);
        color: #ffc107;
        padding: 0.4rem 0.9rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
        border: 1px solid rgba(255, 193, 7, 0.3);
    }

    .series-year {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.875rem;
        font-weight: 500;
    }

    .genre-badge {
        display: inline-block;
        background: rgba(0, 255, 163, 0.1);
        color: #00ffa3;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid rgba(0, 255, 163, 0.2);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }

    .series-description {
        color: #ffffff !important; /* Forces white color */
        font-size: 0.95rem;
        line-height: 1.6;
        margin-top: 1rem;
        flex-grow: 1;
    }

    /* Selectbox styling */
    .stSelectbox > label {
        color: rgba(255, 255, 255, 0.8) !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
    }

    .stSelectbox div[data-baseweb="select"] > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
    }

    /* Alert messages */
    .stSuccess {
        background: rgba(0, 255, 163, 0.1) !important;
        border: 1px solid rgba(0, 255, 163, 0.3) !important;
        border-radius: 12px !important;
        color: #00ffa3 !important;
    }

    .stError {
        background: rgba(255, 82, 82, 0.1) !important;
        border: 1px solid rgba(255, 82, 82, 0.3) !important;
        border-radius: 12px !important;
        color: #ff5252 !important;
    }

    .stWarning {
        background: rgba(255, 193, 7, 0.1) !important;
        border: 1px solid rgba(255, 193, 7, 0.3) !important;
        border-radius: 12px !important;
        color: #ffc107 !important;
    }

    .stInfo {
        background: rgba(0, 212, 255, 0.1) !important;
        border: 1px solid rgba(0, 212, 255, 0.3) !important;
        border-radius: 12px !important;
        color: #00d4ff !important;
    }

    /* Stats badge */
    .stats-badge {
        background: rgba(0, 255, 163, 0.1);
        color: #00ffa3;
        padding: 0.5rem 1.25rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-block;
        border: 1px solid rgba(0, 255, 163, 0.2);
    }

    /* Logout button special styling */
    .logout-btn button {
        background: rgba(255, 255, 255, 0.05) !important;
        color: rgba(255, 255, 255, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: none !important;
    }

    .logout-btn button:hover {
        background: rgba(255, 82, 82, 0.1) !important;
        color: #ff5252 !important;
        border: 1px solid rgba(255, 82, 82, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'users_db' not in st.session_state:
    st.session_state.users_db = {}
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model_data' not in st.session_state:
    st.session_state.model_data = None
if 'watch_history' not in st.session_state:
    st.session_state.watch_history = {}


def hash_password(password):
    """Hash password for storage"""
    return hashlib.sha256(password.encode()).hexdigest()


# ============= MODEL LOADING FUNCTIONS =============

@st.cache_data
def load_dataset():
    """Load and preprocess the dataset"""
    try:
        df = pd.read_csv('TMDB_tv_dataset_v3.csv')

        df.drop_duplicates(subset=['name'], keep='first', inplace=True)

        # Robust overview cleaning
        df['overview'] = df['overview'].fillna('').astype(str)
        df['overview'] = df['overview'].str.strip()
        df['overview'] = df['overview'].replace('', np.nan)
        df.dropna(subset=['overview'], inplace=True)

        def parse_list_string(data_str):
            if pd.isna(data_str): return []
            try:
                return ast.literal_eval(data_str)
            except (ValueError, SyntaxError):
                return []

        df['origin_country'] = df['origin_country'].fillna('[]')
        df['origin_country'] = df['origin_country'].apply(parse_list_string)

        def parse_genres(genre_str):
            if pd.isna(genre_str) or genre_str == '':
                return []
            return genre_str.split(', ')

        df['genres'] = df['genres'].apply(parse_genres)

        # Clean and prepare the 'languages' column (from notebook)
        df['languages'] = df['languages'].fillna('unknown')
        df['languages'] = df['languages'].apply(lambda x: [x])

        df.reset_index(drop=True, inplace=True)

        metadata_cols = ['created_by', 'networks', 'production_companies']
        for col in metadata_cols:
            if col in df.columns:
                df[col] = df[col].apply(parse_list_string)

        for col in metadata_cols:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: [d['name'] for d in x if isinstance(d, dict) and 'name' in d] if isinstance(x,
                                                                                                          list) else []
                )

        df['vote_average'] = df['vote_average'].fillna(0)
        df['vote_count'] = df['vote_count'].fillna(0)

        if 'poster_path' not in df.columns:
            df['poster_path'] = np.nan

        return df
    except FileNotFoundError:
        st.error("❌ Dataset file 'TMDB_tv_dataset_v3.csv' not found. Please make sure it's in the same directory.")
        return None


@st.cache_resource
def build_recommendation_model(df):
    """Build the hybrid recommendation model"""
    try:
        tfidf = TfidfVectorizer(stop_words='english', min_df=2)
        tfidf_matrix = tfidf.fit_transform(df['overview'])

        # Create tags (structural features) using notebook logic
        df['tags'] = df.apply(lambda row:
                              row['genres'] +
                              (row['created_by'] * 5) +
                              (row['networks'] * 3) +
                              row['production_companies'] +
                              row['origin_country'] +
                              (row['languages'] * 5),
                              axis=1)

        mlb = MultiLabelBinarizer()
        binarized_tags = mlb.fit_transform(df['tags'])

        scaler = MinMaxScaler()
        numerical_features = scaler.fit_transform(df[['vote_average', 'vote_count']])

        structured_feature_matrix = np.concatenate(
            [binarized_tags, numerical_features],
            axis=1
        ).astype(np.float32)

        # This is the "Clean List"
        indices = pd.Series(df.index, index=df['name']).drop_duplicates()

        return {
            'tfidf_matrix': tfidf_matrix,
            'structured_matrix': structured_feature_matrix,
            'indices': indices,
            'df': df
        }
    except Exception as e:
        st.error(f"❌ Error building model: {str(e)}")
        return None


# ============= RECOMMENDATION LOGIC =============

def get_recommendations(title, model_data, w_semantic=0.5, w_structural=0.5, top_n=10):
    """
    Get hybrid recommendations for a single series (item-to-item).
    """
    indices = model_data['indices']
    df = model_data['df']
    tfidf_matrix = model_data['tfidf_matrix']
    structured_feature_matrix = model_data['structured_matrix']

    if title not in indices:
        return None

    idx = indices[title]

    semantic_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]
    structural_sim = cosine_similarity(
        structured_feature_matrix[idx].reshape(1, -1),
        structured_feature_matrix
    )[0]

    hybrid_sim_scores = (w_semantic * semantic_sim) + (w_structural * structural_sim)

    sim_scores_enum = sorted(
        list(enumerate(hybrid_sim_scores)),
        key=lambda x: x[1],
        reverse=True
    )

    initial_indices = [i for i, score in sim_scores_enum[1:101]]
    candidates = df.iloc[initial_indices]

    quality_candidates = candidates[candidates['vote_average'] >= 4]
    final_candidates = quality_candidates[quality_candidates['vote_count'] >= 10]

    columns = ['name', 'genres', 'vote_average', 'vote_count', 'overview', 'poster_path']
    return final_candidates[columns].head(top_n)


def get_recommendations_from_history(user_history_list, model_data, w_semantic=0.5, w_structural=0.5, top_n=10):
    """
    Get hybrid recommendations based on a user's entire watch history.
    """
    indices = model_data['indices']
    df = model_data['df']
    tfidf_matrix = model_data['tfidf_matrix']
    structured_feature_matrix = model_data['structured_matrix']

    seen_indices = [indices[title] for title in user_history_list if title in indices]

    if not seen_indices:
        return None

    user_semantic_profile = np.asarray(np.mean(tfidf_matrix[seen_indices], axis=0))
    user_structural_profile = np.mean(structured_feature_matrix[seen_indices], axis=0).reshape(1, -1)

    semantic_sim = cosine_similarity(user_semantic_profile, tfidf_matrix)[0]
    structural_sim = cosine_similarity(user_structural_profile, structured_feature_matrix)[0]

    hybrid_sim_scores = (w_semantic * semantic_sim) + (w_structural * structural_sim)

    sim_scores_enum = sorted(
        list(enumerate(hybrid_sim_scores)),
        key=lambda x: x[1],
        reverse=True
    )

    unseen_indices = []
    for i, score in sim_scores_enum:
        if i not in seen_indices:
            unseen_indices.append(i)
        if len(unseen_indices) >= 100:
            break

    if not unseen_indices:
        return None

    candidates = df.iloc[unseen_indices]

    quality_candidates = candidates[candidates['vote_average'] >= 5]
    final_candidates = quality_candidates[quality_candidates['vote_count'] >= 50]

    columns = ['name', 'genres', 'vote_average', 'vote_count', 'overview', 'poster_path']
    return final_candidates[columns].head(top_n)


def get_top_rated(df, top_n=10):
    """
    Get top-rated shows as a fallback for new users.
    """
    candidates = df[df['vote_count'] >= 500].sort_values('vote_average', ascending=False)
    columns = ['name', 'genres', 'vote_average', 'vote_count', 'overview', 'poster_path']
    return candidates[columns].head(top_n)


# --- REVERTED TO SIMPLE LOGIC (Prevents hiding images) ---
def get_poster_url(poster_path):
    """
    Convert poster_path to full TMDB image URL
    """
    if pd.isna(poster_path) or str(poster_path) == 'nan' or poster_path == '':
        return None
    return f"https://image.tmdb.org/t/p/w500{poster_path}"


# ---------------------------------------------------------

# ============= UI FUNCTIONS =============

def login_page():
    """Modern Login and Signup Page"""
    col1, col2, col3 = st.columns([1, 1.5, 1])

    with col2:
        st.markdown("<div class='brand-logo'>WEB SERIES RECOMMENDER</div>", unsafe_allow_html=True)
        st.markdown("<div class='brand-tagline'>Made by MK2</div>", unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["Sign In", "Create Account"])

        with tab1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            with st.form("login_form", clear_on_submit=False):
                login_username = st.text_input("Username", key="login_user", placeholder="Enter your username")
                login_password = st.text_input("Password", type="password", key="login_pass",
                                               placeholder="Enter your password")
                submitted = st.form_submit_button("Sign In")

                if submitted:
                    if login_username and login_password:
                        hashed_pass = hash_password(login_password)
                        if login_username in st.session_state.users_db:
                            if st.session_state.users_db[login_username] == hashed_pass:
                                st.session_state.logged_in = True
                                st.session_state.username = login_username
                                if login_username not in st.session_state.watch_history:
                                    st.session_state.watch_history[login_username] = []
                                st.success("✓ Welcome back!")
                                st.rerun()
                            else:
                                st.error("✗ Invalid password")
                        else:
                            st.error("✗ Account not found")
                    else:
                        st.warning("⚠ Please fill in all fields")
            st.markdown("</div>", unsafe_allow_html=True)

        with tab2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            with st.form("signup_form", clear_on_submit=True):
                signup_username = st.text_input("Choose Username", key="signup_user",
                                                placeholder="Create a unique username")
                signup_password = st.text_input("Choose Password", type="password", key="signup_pass",
                                                placeholder="Minimum 6 characters")
                signup_password_confirm = st.text_input("Confirm Password", type="password", key="signup_pass_confirm",
                                                        placeholder="Re-enter password")
                submitted = st.form_submit_button("Create Account")

                if submitted:
                    if signup_username and signup_password and signup_password_confirm:
                        if signup_password != signup_password_confirm:
                            st.error("✗ Passwords don't match")
                        elif signup_username in st.session_state.users_db:
                            st.error("✗ Username already taken")
                        elif len(signup_password) < 6:
                            st.error("✗ Password must be at least 6 characters")
                        else:
                            st.session_state.users_db[signup_username] = hash_password(signup_password)
                            st.session_state.watch_history[signup_username] = []
                            st.success("✓ Account created successfully! Please sign in.")
                    else:
                        st.warning("⚠ Please fill in all fields")
            st.markdown("</div>", unsafe_allow_html=True)


def display_recommendation_card(series):
    """
    Renders a single recommendation card using the custom CSS.
    """
    poster_url = get_poster_url(series.get('poster_path'))

    # --- FIXED: Removed onerror, so images are always shown (even if broken) ---
    if poster_url:
        poster_html = f"<div class='series-poster'><img src='{poster_url}' alt='{series['name']} Poster'></div>"
    else:
        poster_html = "<div class='series-poster'></div>"
    # -------------------------------------------------------------------------

    genres_html = ''.join([f"<span class='genre-badge'>{g}</span>" for g in series['genres'][:3]])
    description = (series['overview'][:150] + '...') if len(series['overview']) > 150 else series['overview']

    card_html = f"""
    <div class='series-card'>
        {poster_html}
        <div class='series-content'>
            <div class='series-title'>{series['name']}</div>
            <div class='series-genres'>{genres_html}</div>
            <div class='series-description'>{description}</div>
            <div class='series-meta'>
                <span class='series-rating'>⭐ {series['vote_average']:.1f}</span>
                <span class='series-year'>👥 {int(series['vote_count'])} votes</span>
            </div>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def recommendation_page():
    """Modern Recommendation Page with ML Model"""
    # Header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"""
        <div class='welcome-container'>
            <h1 class='welcome-title'>Hello, {st.session_state.username} 👋</h1>
            <p class='welcome-subtitle'>Ready to find your next obsession?</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='logout-btn'>", unsafe_allow_html=True)
        if st.button("Logout", key="logout_btn"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Load model if not loaded
    if not st.session_state.model_loaded:
        with st.spinner("🔄 Loading AI recommendation model..."):
            df = load_dataset()
            if df is not None:
                model_data = build_recommendation_model(df)
                if model_data is not None:
                    st.session_state.model_data = model_data
                    st.session_state.df = df
                    st.session_state.model_loaded = True
                    st.success("✓ Model loaded successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to build recommendation model")
                    return
            else:
                st.error("❌ Failed to load dataset")
                return

    series_list = sorted(st.session_state.model_data['indices'].index.tolist())
    w_semantic = 0.5
    w_structural = 0.5

    # Tabs for UI
    tab1, tab2, tab3 = st.tabs(["🤖 For You", "🔍 Discover", "📚 Your History"])

    with tab1:
        st.markdown("<h2>Your Personalized Feed</h2>", unsafe_allow_html=True)
        st.markdown(
            "<p style='color: rgba(255, 255, 255, 0.5);'>Recommendations based on your entire watch history.</p>",
            unsafe_allow_html=True)

        current_user = st.session_state.username
        history = st.session_state.watch_history.get(current_user, [])

        recommendations = None
        if not history:
            st.info("Your history is empty. Showing top-rated shows to get you started!")
            recommendations = get_top_rated(st.session_state.df, top_n=10)
        else:
            with st.spinner("Analyzing your unique profile..."):
                recommendations = get_recommendations_from_history(
                    history, st.session_state.model_data, w_semantic, w_structural, top_n=10
                )

        if recommendations is not None and not recommendations.empty:
            cols = st.columns(2)
            for i, (idx, series) in enumerate(recommendations.iterrows()):
                with cols[i % 2]:
                    display_recommendation_card(series)
        else:
            st.warning("Could not generate 'For You' recommendations at this time.")

    with tab2:
        st.markdown("<h2>Discover Similar Shows</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color: rgba(255, 255, 255, 0.5);'>Find shows similar to one you already love.</p>",
                    unsafe_allow_html=True)

        col1, col2 = st.columns([3, 1])
        with col1:
            selected_series = st.selectbox(
                "What series did you watch?",
                options=[""] + series_list,
                format_func=lambda x: "Select a series..." if x == "" else x,
                key="discover_select"
            )

        with col2:
            st.markdown('<div style="height: 28px;"></div>', unsafe_allow_html=True)
            recommend_btn = st.button("Get Recommendations", type="primary", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        if recommend_btn and selected_series:
            with st.spinner("🤖 AI is analyzing and generating recommendations..."):
                recommendations = get_recommendations(
                    selected_series, st.session_state.model_data, w_semantic, w_structural, top_n=12
                )

                if recommendations is not None and not recommendations.empty:
                    current_user = st.session_state.username
                    if selected_series not in st.session_state.watch_history[current_user]:
                        st.session_state.watch_history[current_user].append(selected_series)

                    st.markdown("<div style='margin: 3rem 0 1.5rem 0;'></div>", unsafe_allow_html=True)
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"<h2>Because you watched '{selected_series}'</h2>", unsafe_allow_html=True)
                    with col2:
                        st.markdown(
                            f"<div class='stats-badge' style='margin-top: 2.5rem;'>{len(recommendations)} Recommendations</div>",
                            unsafe_allow_html=True)

                    cols = st.columns(2)
                    for i, (idx, series) in enumerate(recommendations.iterrows()):
                        with cols[i % 2]:
                            display_recommendation_card(series)
                else:
                    st.warning("⚠️ No recommendations found for this series. Try another one!")

        elif recommend_btn and not selected_series:
            st.warning("⚠️ Please select a series first!")

    with tab3:
        st.markdown("<h2>📚 Your Watch History</h2>", unsafe_allow_html=True)

        history = st.session_state.watch_history.get(st.session_state.username, [])

        if history:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            history_html = ''.join([f"<span class='genre-badge'>{series}</span>" for series in reversed(history[-20:])])
            st.markdown(history_html, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
            if st.button("Clear Watch History", key="clear_history"):
                st.session_state.watch_history[st.session_state.username] = []
                st.success("History cleared!")
                st.rerun()
        else:
            st.info("You haven't searched for any shows yet. Use the 'Discover' tab to build your history.")


# Main app
def main():
    if st.session_state.logged_in:
        recommendation_page()
    else:
        login_page()


if __name__ == "__main__":
    main()