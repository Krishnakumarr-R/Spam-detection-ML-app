import streamlit as st
import joblib
import pickle

# Set page configuration
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Cards */
    .stAlert {
        border-radius: 15px;
        border-left: 5px solid;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.5rem 2rem;
        transition: all 0.3s;
    }
    
    /* Text area */
    .stTextArea>div>div>textarea {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
    }
    
    /* Metrics */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Example cards */
    .example-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .example-card:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Header */
    .main-header {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    </style>
""", unsafe_allow_html=True)

# Load the model and vectorizer
@st.cache_resource
def load_model():
    try:
        model = joblib.load('spam_classifier_model.joblib')
        vectorizer = joblib.load('tfidf_vectorizer.joblib')
        
        if not (hasattr(vectorizer, '_tfidf') and hasattr(vectorizer._tfidf, 'idf_')):
            st.error("❌ Vectorizer is not fitted! Please retrain the model.")
            return None, None
            
        return model, vectorizer
        
    except Exception as joblib_error:
        try:
            with open('spam_classifier_model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('tfidf_vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
            
            if not (hasattr(vectorizer, '_tfidf') and hasattr(vectorizer._tfidf, 'idf_')):
                st.error("❌ Vectorizer is not fitted! Please retrain the model.")
                return None, None
                
            return model, vectorizer
            
        except FileNotFoundError:
            st.error("⚠️ Model files not found! Please ensure model files are in the same directory.")
            return None, None
        except Exception as pickle_error:
            st.error(f"❌ Error loading model files. Try: pip install --upgrade numpy scikit-learn joblib")
            return None, None

model, vectorizer = load_model()

# Example emails
SPAM_EXAMPLES = {
    "🎁 Prize Winner": "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.",
    "💰 Urgent Money": "URGENT! You've been selected for a $5000 cash prize! Click here immediately to claim before it expires. This is not a scam!",
    "📱 Suspicious Link": "Free Msg: Ringtone!From: http://tms. widelive.com/index. wml?id=1b6a5ecef91ff9*37819&first=true18:0430-JUL-05",
    "🎰 Lottery Scam": "YOU WON THE LOTTERY! £1,000,000 waiting for you. Send your bank details to claim. Reply ASAP before offer expires!"
}

HAM_EXAMPLES = {
    "📅 Meeting Invite": "Hi John, just wanted to confirm our meeting tomorrow at 3 PM in conference room B. Please bring the quarterly reports. See you then!",
    "☕ Friend Message": "Hey! How are you doing? Want to grab coffee this weekend? Let me know what works for you. Would love to catch up!",
    "💼 Work Email": "Dear team, please review the attached documents before our meeting on Friday. Let me know if you have any questions. Thanks!",
    "👨‍👩‍👧 Family Update": "Hi everyone! Just wanted to share some photos from our vacation. Hope you're all doing well. Let's plan a family dinner soon!"
}

# Header
st.markdown("""
    <div class="main-header">
        <h1 style="margin:0; color: #667eea;">🛡️ Email Spam Classifier</h1>
        <p style="margin:0; color: #666; font-size: 1.1rem;">AI-powered spam detection using Machine Learning</p>
    </div>
""", unsafe_allow_html=True)

# Create layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📧 Enter Email Content")
    
    # Initialize session state for input
    if 'email_input' not in st.session_state:
        st.session_state.email_input = ""
    
    user_input = st.text_area(
        "Paste or type your email message here:",
        value=st.session_state.email_input,
        height=250,
        placeholder="Example: WINNER!! You have been selected to receive a prize...",
    )
    
    col_btn1, col_btn2 = st.columns([3, 1])
    
    with col_btn1:
        classify_button = st.button("🔍 Classify Email", type="primary", use_container_width=True)
    
    with col_btn2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.email_input = ""
            st.rerun()
    
    # Classification logic
    if classify_button:
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text to classify!")
        elif model is None or vectorizer is None:
            st.error("❌ Model not loaded. Cannot make predictions.")
        else:
            with st.spinner("🔄 Analyzing email..."):
                # Transform input text
                input_features = vectorizer.transform([user_input])
                
                # Make prediction
                prediction = model.predict(input_features)
                prediction_proba = model.predict_proba(input_features)
                
                # Display results
                st.markdown("---")
                
                if prediction[0] == 1:
                    # HAM (Legitimate)
                    st.success("### ✅ LEGITIMATE EMAIL (Ham Mail)")
                    st.markdown("**This email appears to be safe and legitimate.**")
                    confidence = prediction_proba[0][1] * 100
                    
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric("📊 Confidence", f"{confidence:.1f}%")
                    with col_m2:
                        st.metric("✅ Ham Probability", f"{prediction_proba[0][1]*100:.1f}%")
                    with col_m3:
                        st.metric("🚨 Spam Probability", f"{prediction_proba[0][0]*100:.1f}%")
                    
                    # Probability bars
                    st.markdown("#### Probability Distribution")
                    st.markdown("**Legitimate (Ham)**")
                    st.progress(prediction_proba[0][1])
                    st.markdown("**Spam**")
                    st.progress(prediction_proba[0][0])
                    
                    st.balloons()
                    
                else:
                    # SPAM
                    st.error("### 🚨 SPAM EMAIL DETECTED!")
                    st.markdown("**⚠️ This email appears to be spam or malicious. Be cautious!**")
                    confidence = prediction_proba[0][0] * 100
                    
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric("📊 Confidence", f"{confidence:.1f}%")
                    with col_m2:
                        st.metric("🚨 Spam Probability", f"{prediction_proba[0][0]*100:.1f}%")
                    with col_m3:
                        st.metric("✅ Ham Probability", f"{prediction_proba[0][1]*100:.1f}%")
                    
                    # Probability bars
                    st.markdown("#### Probability Distribution")
                    st.markdown("**Spam**")
                    st.progress(prediction_proba[0][0])
                    st.markdown("**Legitimate (Ham)**")
                    st.progress(prediction_proba[0][1])
    
    # Update session state when text area changes
    if user_input != st.session_state.email_input:
        st.session_state.email_input = user_input

# Sidebar with examples
with col2:
    st.markdown("### 🚨 Spam Examples")
    st.markdown("*Click to load example*")
    
    for title, text in SPAM_EXAMPLES.items():
        with st.container():
            if st.button(title, key=f"spam_{title}", use_container_width=True):
                st.session_state.email_input = text
                st.rerun()
            with st.expander("Preview"):
                st.caption(text[:100] + "...")
    
    st.markdown("---")
    st.markdown("### ✅ Legitimate Examples")
    st.markdown("*Click to load example*")
    
    for title, text in HAM_EXAMPLES.items():
        with st.container():
            if st.button(title, key=f"ham_{title}", use_container_width=True):
                st.session_state.email_input = text
                st.rerun()
            with st.expander("Preview"):
                st.caption(text[:100] + "...")

# Sidebar information
with st.sidebar:
    st.markdown("## ℹ️ About")
    st.info("""
        This app uses **Machine Learning** to classify emails as:
        - **Spam**: Unwanted or suspicious emails
        - **Ham**: Legitimate emails
        
        **Model**: Logistic Regression  
        **Features**: TF-IDF Vectorization
    """)
    
    st.markdown("## 🎯 Spam Indicators")
    st.markdown("""
        - 🎁 Prize/money claims
        - 🔗 Suspicious links
        - ⚠️ Too good to be true offers
        - 📝 Poor grammar
        - 🔐 Requests for personal info
        - ⏰ Urgent time pressure
    """)
    
    st.markdown("## 🛠️ How to Use")
    st.markdown("""
        1. **Enter** or **paste** email content
        2. **Click** example buttons to auto-fill
        3. **Press** "Classify Email" button
        4. **View** the prediction result
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    if model is not None and vectorizer is not None:
        st.success("✅ Model Loaded")
        st.caption(f"Vocabulary: {len(vectorizer.vocabulary_)} words")
    else:
        st.error("❌ Model Not Loaded")
    
    st.markdown("---")
    st.caption("Built with Streamlit & Scikit-learn")