import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu

# Configuration
API_BASE_URL = "http://localhost:8000"
st.set_page_config(page_title="AI Platform Dashboard", layout="wide", page_icon="🤖")

# Session state for authentication
if 'token' not in st.session_state:
    st.session_state.token = None
if 'user' not in st.session_state:
    st.session_state.user = None

# API Helper Functions
def api_request(endpoint, method="GET", data=None, auth=True):
    headers = {}
    if auth and st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    
    url = f"{API_BASE_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers)
        elif method == "PUT":
            response = requests.put(url, json=data, headers=headers)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None

# Authentication
def login_page():
    st.title("🤖 Enterprise AI Platform")
    st.subheader("Login")
    
    # Check if backend is running
    try:
        health_check = requests.get(f"{API_BASE_URL}/health", timeout=2)
        backend_running = health_check.status_code == 200
    except:
        backend_running = False
    
    if not backend_running:
        st.warning("⚠️ Backend API is not running. Using demo mode.")
        st.info("""
        **To start the backend:**
        ```bash
        cd enterprise-ai-platform
        uvicorn app.main:app --reload
        ```
        """)
        
        # Demo mode login
        with st.form("login_form"):
            username = st.text_input("Username", value="demo")
            password = st.text_input("Password", type="password", value="demo")
            submit = st.form_submit_button("Login (Demo Mode)")
            
            if submit:
                st.session_state.token = "demo_token"
                st.session_state.user = {"username": username, "role": "admin"}
                st.success("Logged in to demo mode!")
                st.rerun()
    else:
        with st.form("login_form"):
            email = st.text_input("Email", value="admin@example.com")
            password = st.text_input("Password", type="password", value="admin123")
            submit = st.form_submit_button("Login")
            
            if submit:
                response = api_request("/api/auth/login", method="POST", 
                                     data={"email": email, "password": password}, 
                                     auth=False)
                if response:
                    st.session_state.token = response.get("access_token")
                    st.session_state.user = {"username": email.split("@")[0], "role": "admin"}
                    st.success("Login successful!")
                    st.rerun()

# Dashboard Page
def dashboard_page():
    st.title("📊 Dashboard")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Models", "12", "+2")
    with col2:
        st.metric("Active Users", "48", "+5")
    with col3:
        st.metric("API Calls Today", "1,234", "+15%")
    with col4:
        st.metric("Avg Response Time", "245ms", "-12ms")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Performance")
        data = pd.DataFrame({
            'Model': ['GPT-4', 'Claude', 'Llama-2', 'Custom'],
            'Accuracy': [95, 93, 88, 91]
        })
        fig = px.bar(data, x='Model', y='Accuracy', color='Accuracy')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("API Usage (Last 7 Days)")
        dates = pd.date_range(end=datetime.now(), periods=7)
        usage_data = pd.DataFrame({
            'Date': dates,
            'Requests': [850, 920, 1100, 980, 1050, 1150, 1234]
        })
        fig = px.line(usage_data, x='Date', y='Requests')
        st.plotly_chart(fig, use_container_width=True)

# Models Page
def models_page():
    st.title("🤖 AI Models")
    
    tab1, tab2 = st.tabs(["Active Models", "Deploy New Model"])
    
    with tab1:
        models_data = {
            'Name': ['GPT-4 Turbo', 'Claude 3', 'Llama-2-70B', 'Custom RAG'],
            'Type': ['LLM', 'LLM', 'LLM', 'RAG'],
            'Status': ['Active', 'Active', 'Active', 'Active'],
            'Requests': [450, 320, 180, 284]
        }
        df = pd.DataFrame(models_data)
        st.dataframe(df, use_container_width=True)
    
    with tab2:
        with st.form("deploy_model"):
            model_name = st.text_input("Model Name")
            model_type = st.selectbox("Model Type", ["LLM", "ML", "RAG", "Custom"])
            model_file = st.file_uploader("Upload Model File")
            
            if st.form_submit_button("Deploy Model"):
                st.success(f"Model '{model_name}' deployed successfully!")

# Predictions Page
def predictions_page():
    st.title("🔮 Predictions")
    
    tab1, tab2 = st.tabs(["Make Prediction", "History"])
    
    with tab1:
        st.subheader("Run Prediction")
        
        model = st.selectbox("Select Model", ["GPT-4 Turbo", "Claude 3", "Custom RAG"])
        
        input_type = st.radio("Input Type", ["Text", "File"])
        
        if input_type == "Text":
            user_input = st.text_area("Enter your input", height=150)
        else:
            uploaded_file = st.file_uploader("Upload file")
        
        if st.button("Run Prediction", type="primary"):
            with st.spinner("Processing..."):
                st.success("Prediction completed!")
                st.json({
                    "prediction": "Sample prediction result",
                    "confidence": 0.95,
                    "processing_time": "1.2s"
                })
    
    with tab2:
        st.subheader("Prediction History")
        history_data = {
            'Timestamp': [datetime.now() - timedelta(hours=i) for i in range(5)],
            'Model': ['GPT-4', 'Claude', 'GPT-4', 'Custom RAG', 'Llama-2'],
            'Status': ['Success', 'Success', 'Success', 'Failed', 'Success'],
            'Time': ['1.2s', '0.8s', '1.5s', '-', '2.1s']
        }
        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True)

# LLM Chat Page
def llm_chat_page():
    st.title("💬 LLM Chat")
    
    model = st.selectbox("Select Model", ["GPT-4 Turbo", "Claude 3 Opus", "Llama-2-70B"])
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            response = f"Response from {model}: This is a sample response to '{prompt}'"
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# RAG System Page
def rag_page():
    st.title("📚 RAG System")
    
    tab1, tab2 = st.tabs(["Query Documents", "Manage Documents"])
    
    with tab1:
        st.subheader("Query Your Documents")
        query = st.text_input("Enter your question")
        
        if st.button("Search", type="primary"):
            with st.spinner("Searching..."):
                st.success("Found relevant documents!")
                st.markdown("**Answer:** Sample answer based on your documents...")
                
                with st.expander("View Sources"):
                    st.markdown("- Document 1: page 5")
                    st.markdown("- Document 2: page 12")
    
    with tab2:
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True)
        
        if st.button("Process Documents"):
            st.success(f"Processed {len(uploaded_files) if uploaded_files else 0} documents")
        
        st.subheader("Document Library")
        docs_data = {
            'Name': ['Product Manual.pdf', 'API Docs.pdf', 'User Guide.pdf'],
            'Size': ['2.5 MB', '1.8 MB', '3.2 MB'],
            'Uploaded': ['2 days ago', '5 days ago', '1 week ago']
        }
        st.dataframe(pd.DataFrame(docs_data), use_container_width=True)

# Users Page
def users_page():
    st.title("👥 Users")
    
    tab1, tab2 = st.tabs(["User List", "Add User"])
    
    with tab1:
        users_data = {
            'Username': ['admin', 'john_doe', 'jane_smith', 'ai_engineer'],
            'Email': ['admin@company.com', 'john@company.com', 'jane@company.com', 'engineer@company.com'],
            'Role': ['Admin', 'User', 'User', 'Developer'],
            'Status': ['Active', 'Active', 'Active', 'Active']
        }
        df = pd.DataFrame(users_data)
        st.dataframe(df, use_container_width=True)
    
    with tab2:
        with st.form("add_user"):
            username = st.text_input("Username")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            role = st.selectbox("Role", ["User", "Developer", "Admin"])
            
            if st.form_submit_button("Create User"):
                st.success(f"User '{username}' created successfully!")

# Settings Page
def settings_page():
    st.title("⚙️ Settings")
    
    tab1, tab2, tab3 = st.tabs(["General", "API Keys", "Monitoring"])
    
    with tab1:
        st.subheader("General Settings")
        st.text_input("Platform Name", value="Enterprise AI Platform")
        st.number_input("Max Concurrent Requests", value=100)
        st.number_input("Request Timeout (seconds)", value=30)
        st.button("Save Settings", type="primary")
    
    with tab2:
        st.subheader("API Keys")
        st.text_input("OpenAI API Key", type="password", value="sk-...")
        st.text_input("Anthropic API Key", type="password", value="sk-ant-...")
        st.button("Update Keys", type="primary")
    
    with tab3:
        st.subheader("Monitoring")
        st.checkbox("Enable Prometheus Metrics", value=True)
        st.checkbox("Enable Request Logging", value=True)
        st.checkbox("Enable Error Tracking", value=True)
        st.button("Save Monitoring Settings", type="primary")

# Main App
def main():
    if not st.session_state.token:
        login_page()
    else:
        # Sidebar Navigation
        with st.sidebar:
            st.title("🤖 AI Platform")
            st.write(f"Welcome, {st.session_state.user.get('username', 'User') if st.session_state.user else 'User'}")
            
            selected = option_menu(
                menu_title=None,
                options=["Dashboard", "Models", "Predictions", "LLM Chat", "RAG System", "Users", "Settings"],
                icons=["speedometer2", "robot", "graph-up", "chat-dots", "book", "people", "gear"],
                default_index=0,
            )
            
            st.divider()
            if st.button("Logout"):
                st.session_state.token = None
                st.session_state.user = None
                st.rerun()
        
        # Page Routing
        if selected == "Dashboard":
            dashboard_page()
        elif selected == "Models":
            models_page()
        elif selected == "Predictions":
            predictions_page()
        elif selected == "LLM Chat":
            llm_chat_page()
        elif selected == "RAG System":
            rag_page()
        elif selected == "Users":
            users_page()
        elif selected == "Settings":
            settings_page()

if __name__ == "__main__":
    main()
