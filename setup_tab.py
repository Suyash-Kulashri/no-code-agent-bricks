import streamlit as st
from databricks.sdk import WorkspaceClient
from utils import clear_current_step_messages, create_catalog_and_schema

def render_setup_tab(workspace_url, token):
    """Render the Setup tab"""
    st.subheader("Catalog and Schema Setup")
    
    with st.form("catalog_schema_form"):
        col1, col2 = st.columns(2)
        with col1:
            catalog_name = st.text_input("Catalog Name", value="ai_agent_catalog", placeholder="my_catalog")
            schema_name = st.text_input("Schema Name", value="default_schema", placeholder="my_schema")
        with col2:
            storage_location = st.text_input(
                "Storage Location", 
                placeholder="s3://my-bucket/databricks",
                help="Cloud storage location for the catalog"
            )
        
        create_infra = st.form_submit_button("🏗️ Create Infrastructure", use_container_width=True)
    
    if create_infra and workspace_url and token and catalog_name and schema_name and storage_location:
        clear_current_step_messages()
        if workspace_url.startswith("https://"):
            try:
                client = WorkspaceClient(host=workspace_url, token=token)
                create_catalog_and_schema(client, catalog_name, schema_name, storage_location)
            except Exception as e:
                st.error(f"❌ Connection error: {str(e)}")
        else:
            st.error("❌ Please provide a valid workspace URL starting with https://")
