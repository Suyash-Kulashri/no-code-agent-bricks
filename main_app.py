import streamlit as st
from databricks.sdk import WorkspaceClient
import os
from dotenv import load_dotenv

# Import tab modules
from setup_tab import render_setup_tab
from data_upload_tab import render_data_upload_tab
from document_parsing_tab import render_document_parsing_tab  # New import
from vectorization_tab import render_vectorization_tab
from agent_creation_tab import render_agent_creation_tab
from chat_tab import render_chat_tab

# Load environment variables
load_dotenv()

# Streamlit app configuration
st.set_page_config(page_title="Databricks Agent Brick Creator", layout="wide")
st.title("🧱 Databricks No-Code Agent Brick Creator")

# Initialize session state
if "status_messages" not in st.session_state:
    st.session_state.status_messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "parsed_tables" not in st.session_state:  # New session state for parsed tables
    st.session_state.parsed_tables = []
if "vectorized_indexes" not in st.session_state:
    st.session_state.vectorized_indexes = []
if "agent_endpoints" not in st.session_state:
    st.session_state.agent_endpoints = []
if "current_step_messages" not in st.session_state:
    st.session_state.current_step_messages = []


workspace_url = os.getenv("DATABRICKS_HOST")
token = os.getenv("DATABRICKS_TOKEN")
# Sidebar for Databricks connection
# with st.sidebar:
#     st.header("🔗 Databricks Connection")
#     workspace_url = os.getenv("DATABRICKS_HOST") or st.text_input(
#         "Databricks Workspace URL", 
#         placeholder="https://<your-databricks-instance>",
#         help="Your Databricks workspace URL"
#     )
#     token = os.getenv("DATABRICKS_TOKEN") or st.text_input(
#         "Databricks Personal Access Token", 
#         type="password",
#         help="Your Databricks PAT for authentication"
#     )
    
# Connection status
with st.sidebar:
    st.header("🔗 Databricks Connection")
    if workspace_url and token:
            try:
                if workspace_url.startswith("https://"):
                    client = WorkspaceClient(host=workspace_url, token=token)
                    st.success("✅ Connection configured")
                else:
                    st.error("❌ Invalid URL format")
            except Exception as e:
                st.error(f"❌ Connection error: {str(e)}")

# Main content tabs - Updated to include Document Parsing
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏗️ Setup", 
    "📁 Data Upload", 
    "📄 Document Parsing",  # New tab
    "🔍 Vectorization", 
    "🤖 Agent Creation", 
    "💬 Chat with Agent"
])

# Render each tab
with tab1:
    render_setup_tab(workspace_url, token)

with tab2:
    render_data_upload_tab(workspace_url, token)

with tab3:
    render_document_parsing_tab()  # New tab rendering

with tab4:
    render_vectorization_tab(workspace_url, token)

with tab5:
    render_agent_creation_tab(workspace_url, token)

with tab6:
    render_chat_tab(workspace_url, token)

# Status Messages Display (only current step)
if st.session_state.current_step_messages:
    st.divider()
    st.subheader("📋 Current Step Status")
    col1, col2 = st.columns([3, 1])
    with col1:
        for msg_info in st.session_state.current_step_messages:
            if msg_info['is_success']:
                st.success(msg_info['message'])
            else:
                st.error(msg_info['message'])
    with col2:
        if st.button("🗑️ Clear Logs", key="clear_status_logs"):
            st.session_state.current_step_messages = []
            st.rerun()

# Summary Dashboard
if any([st.session_state.uploaded_files, st.session_state.parsed_tables, 
        st.session_state.vectorized_indexes, st.session_state.agent_endpoints]):
    
    st.divider()
    st.subheader("📊 System Overview")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📁 Uploaded Files",
            value=len(st.session_state.uploaded_files),
            help="Files uploaded to Databricks volumes"
        )
    
    with col2:
        st.metric(
            label="📄 Parsed Tables",
            value=len(st.session_state.parsed_tables),
            help="Tables created from parsed documents"
        )
    
    with col3:
        st.metric(
            label="🔍 Vector Indexes",
            value=len(st.session_state.vectorized_indexes),
            help="Vector search indexes created"
        )
    
    with col4:
        st.metric(
            label="🤖 AI Agents",
            value=len(st.session_state.agent_endpoints),
            help="Agent endpoints created"
        )

# Recent Activity Section
if st.session_state.parsed_tables or st.session_state.vectorized_indexes:
    st.subheader("📋 Recent Activity")
    
    # Show recent parsed tables
    if st.session_state.parsed_tables:
        st.write("**📄 Recently Parsed Tables:**")
        for table in st.session_state.parsed_tables[-3:]:  # Show last 3
            st.write(f"• `{table['full_name']}` - {table['file_count']} files processed")
    
    # Show recent vector indexes
    if st.session_state.vectorized_indexes:
        st.write("**🔍 Recently Created Vector Indexes:**")
        for index in st.session_state.vectorized_indexes[-3:]:  # Show last 3
            st.write(f"• `{index['name']}` - Status: {index.get('status', 'Unknown')}")

# Agent Endpoints Display (existing)
if st.session_state.agent_endpoints:
    st.subheader("🤖 Created Agents")
    for agent in st.session_state.agent_endpoints:
        with st.expander(f"🤖 {agent['name']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Model:** {agent['model']}")
                st.write(f"**Vector Index:** {agent.get('vector_index', 'None')}")
                st.write(f"**Status:** {agent.get('status', 'Unknown')}")
            with col2:
                st.code(agent['url'], language="text")

# Debug panel (collapsible) - Updated
with st.expander("🔧 Debug Information"):
    from utils import clear_current_step_messages, add_status_message, get_catalogs, get_serving_endpoints
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Session State:**")
        st.write(f"Files: {len(st.session_state.uploaded_files)}")
        st.write(f"Parsed Tables: {len(st.session_state.parsed_tables)}")  # New
        st.write(f"Indexes: {len(st.session_state.vectorized_indexes)}")
        st.write(f"Agents: {len(st.session_state.agent_endpoints)}")
    
    with col2:
        if st.button("🗑️ Clear All Data"):
            st.session_state.uploaded_files = []
            st.session_state.parsed_tables = []  # New
            st.session_state.vectorized_indexes = []
            st.session_state.agent_endpoints = []
            st.session_state.current_step_messages = []
            st.rerun()
    
    with col3:
        if st.button("📊 System Check"):
            clear_current_step_messages()
            if workspace_url and token:
                try:
                    client = WorkspaceClient(host=workspace_url, token=token)
                    
                    # Check system components
                    catalogs = get_catalogs(client)
                    add_status_message(f"✅ Found {len(catalogs)} catalogs")
                    
                    endpoints = get_serving_endpoints(client)
                    add_status_message(f"✅ Found {len(endpoints)} serving endpoints")
                    
                    try:
                        warehouses = list(client.warehouses.list())
                        running = [w for w in warehouses if w.state.value == "RUNNING"]
                        add_status_message(f"✅ Found {len(running)} running SQL warehouses")
                    except:
                        add_status_message("⚠️ Could not check SQL warehouses")
                    
                    # Check for parsed tables
                    if st.session_state.parsed_tables:
                        add_status_message(f"✅ Found {len(st.session_state.parsed_tables)} parsed tables in session")
                    
                except Exception as e:
                    add_status_message(f"❌ System check failed: {str(e)}", False)
            else:
                add_status_message("❌ Configure connection first", False)

# Footer with improvements - Updated
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    ### 📋 Enhanced Workflow:
    1. **Setup**: Configure Databricks connection and create catalog/schema
    2. **Upload**: Upload multiple CSV/PDF/text files to selected catalog/schema
    3. **Parse**: Parse documents into structured tables with chunking and metadata
    4. **Vectorize**: Create vector indexes from parsed tables (batch processing available)
    5. **Create Agent**: Build AI agents with Databricks or external models
    6. **Chat**: Interact with your agents through the chat interface
    """)
with col2:
    st.markdown("""
    ### 🔧 System Requirements:
    - ✅ Databricks workspace with Unity Catalog
    - ✅ Vector Search enabled
    - ✅ Model Serving capabilities
    - ✅ SQL Warehouse for table operations
    - ✅ File processing capabilities
    """)

st.markdown("---")
st.markdown("""
### 🎉 Latest Features - Document Parsing Integration:

**✅ Advanced Document Parsing:**
- Parse PDF, TXT, DOCX, CSV, and JSON files into structured tables
- Intelligent text chunking with configurable size and overlap
- Metadata preservation (file paths, types, word counts, page numbers)
- Batch processing with progress tracking

**✅ Smart Content Processing:**
- PDF page-aware parsing with text extraction
- DOCX paragraph-level processing
- CSV row-by-row content structuring
- JSON hierarchical content flattening
- Text file intelligent chunking

**✅ Enhanced Data Pipeline:**
- **Upload** → **Parse** → **Vectorize** → **Agent Creation** workflow
- Parsed tables ready for vector search indexing
- Change Data Feed (CDF) enabled tables for incremental updates
- Table summary and statistics tracking

**✅ Improved User Experience:**
- File type grouping and size summaries
- Parsing preview functionality
- Progress tracking for large document sets
- Session state management for parsed tables
- Integrated workflow across all tabs

**🎯 Production Ready Document Processing Pipeline!**
""")