import streamlit as st
from databricks.sdk import WorkspaceClient
from agent_creation_utils import (
    get_catalogs, get_schemas, get_foundation_models, create_agent_endpoint,
    clear_current_step_messages, add_status_message, check_endpoint_status,
    test_endpoint_chat, list_vector_search_indexes, get_available_serving_endpoints,
    delete_endpoint
)

def render_agent_creation_tab(workspace_url, token):
    """Render the improved Agent Creation tab"""
    st.header("🤖 AI Agent Creation")
    
    if not (workspace_url and token):
        st.warning("⚠️ Please configure Databricks connection first")
        return
    
    try:
        client = WorkspaceClient(host=workspace_url, token=token)
        
        # Show existing endpoints
        _show_existing_endpoints(client)
        
        st.divider()
        
        # Agent creation section
        st.subheader("Create New Agent")
        
        # Quick tips
        with st.expander("💡 Quick Tips", expanded=False):
            st.markdown("""
            **Before creating agents:**
            - Ensure you have permissions to create serving endpoints
            - For external models, have your API key ready
            - Vector indexes are optional but enhance capability with RAG
            - Deployment typically takes 5-10 minutes
            
            **Naming conventions:**
            - Use descriptive names like `customer_support_agent`
            - No spaces - use underscores instead
            - Keep names unique and meaningful
            """)
        
        # Agent type selection
        agent_type = st.radio(
            "**Agent Type**",
            ["Databricks Foundation Model", "External Model (OpenAI/Anthropic/etc.)"],
            horizontal=True,
            help="Choose between using Databricks hosted models or external API models"
        )
        
        if agent_type == "Databricks Foundation Model":
            _render_databricks_agent_form(client)
        else:
            _render_external_agent_form(client)
            
    except Exception as e:
        st.error(f"❌ Error initializing Databricks client: {str(e)}")
        st.info("💡 Check your workspace URL and token configuration")

def _show_existing_endpoints(client: WorkspaceClient):
    """Show existing endpoints"""
    # Show user-created endpoints
    if hasattr(st.session_state, 'custom_endpoints') and st.session_state.custom_endpoints:
        st.subheader("🔧 Your Created Endpoints")
        
        for i, endpoint in enumerate(st.session_state.custom_endpoints):
            with st.expander(f"🤖 {endpoint['name']} - {endpoint['status']}", expanded=False):
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    st.write(f"**Name:** {endpoint['name']}")
                    st.write(f"**Model:** {endpoint['model']}")
                    st.write(f"**Status:** {endpoint['status']}")
                    st.write(f"**Type:** {'External' if endpoint.get('is_external', False) else 'Databricks'}")
                    if endpoint.get('vector_indexes'):
                        st.write(f"**Vector Indexes:** {len(endpoint['vector_indexes'])}")
                
                with col2:
                    if st.button("🔄 Status", key=f"status_{i}"):
                        try:
                            status_info = check_endpoint_status(client, endpoint['endpoint_name'])
                            endpoint['status'] = status_info.get('status', 'Unknown')
                            st.json(status_info)
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                
                with col3:
                    if st.button("💬 Test", key=f"test_{i}"):
                        st.session_state[f'show_test_{i}'] = not st.session_state.get(f'show_test_{i}', False)
                
                with col4:
                    if st.button("🗑️ Delete", key=f"delete_{i}", type="secondary"):
                        if st.session_state.get(f'confirm_delete_{i}', False):
                            try:
                                if delete_endpoint(client, endpoint['endpoint_name']):
                                    st.session_state.custom_endpoints.remove(endpoint)
                                    st.success("✅ Endpoint deleted")
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting: {str(e)}")
                        else:
                            st.session_state[f'confirm_delete_{i}'] = True
                            st.warning("Click again to confirm")
                
                # Test chat interface
                if st.session_state.get(f'show_test_{i}', False):
                    st.markdown("**Test Chat:**")
                    test_message = st.text_input(
                        "Message:", 
                        value="Hello! Can you help me?",
                        key=f"test_msg_{i}"
                    )
                    
                    if st.button("Send", key=f"send_{i}"):
                        with st.spinner("Sending message..."):
                            try:
                                response = test_endpoint_chat(
                                    client, endpoint['endpoint_name'], 
                                    test_message, endpoint.get('system_prompt', '')
                                )
                                st.markdown("**Response:**")
                                st.write(response)
                            except Exception as e:
                                st.error(f"Test failed: {str(e)}")
    
    # Show all available endpoints
    with st.expander("📋 All Serving Endpoints in Workspace", expanded=False):
        try:
            endpoints = get_available_serving_endpoints(client)
            if endpoints:
                for endpoint in endpoints:
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"**{endpoint['name']}**")
                    with col2:
                        st.write(f"Status: {endpoint['state']}")
                    with col3:
                        st.write(f"Update: {endpoint['config_update']}")
            else:
                st.info("No serving endpoints found")
        except Exception as e:
            st.error(f"Error loading endpoints: {str(e)}")

def _render_databricks_agent_form(client: WorkspaceClient):
    """Render Databricks foundation model agent creation form"""
    st.subheader("🏢 Databricks Foundation Model Agent")
    
    # Load available models
    with st.spinner("Loading available models..."):
        foundation_models = get_foundation_models(client)
    
    if not foundation_models:
        st.error("❌ No foundation models available. Check your workspace configuration.")
        return
    
    # Vector index selection
    st.markdown("**Knowledge Base Configuration (Optional):**")
    vector_indexes = _render_vector_index_selection(client, "db")
    
    # Main form
    with st.form("databricks_agent_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        
        with col1:
            agent_name = st.text_input(
                "Agent Name *", 
                value="my_knowledge_agent",
                help="Unique name for your agent (no spaces, use underscores)",
                placeholder="e.g., customer_support_agent"
            )
            
            foundation_model = st.selectbox(
                "Foundation Model *",
                options=foundation_models,
                help="Select a Databricks foundation model for chat/instruction following",
                index=0 if foundation_models else None
            )
        
        with col2:
            agent_description = st.text_area(
                "Agent Description",
                placeholder="Describe what your agent does and its purpose...",
                help="This helps you identify the agent later"
            )
            
            max_tokens = st.slider(
                "Max Response Tokens",
                min_value=100,
                max_value=2000,
                value=800,
                help="Maximum length of agent responses"
            )
        
        system_prompt = st.text_area(
            "System Prompt *",
            value="""You are a helpful AI assistant. Answer questions accurately and concisely. If you don't know the answer, say so clearly. Be professional and helpful in your responses.""",
            height=120,
            help="Define how your agent should behave and respond to users",
            max_chars=1000
        )
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            temperature = st.slider(
                "Temperature (Creativity)",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Higher values make responses more creative but less predictable"
            )
            
            enable_streaming = st.checkbox(
                "Enable Response Streaming",
                value=True,
                help="Stream responses word by word for better user experience"
            )
        
        create_databricks_agent = st.form_submit_button(
            "🚀 Create Databricks Agent", 
            use_container_width=True,
            type="primary"
        )
    
    if create_databricks_agent:
        if not _validate_agent_form(agent_name, foundation_model, system_prompt):
            return
        
        clear_current_step_messages()
        
        with st.spinner("Creating Databricks agent endpoint..."):
            try:
                endpoint_url = create_agent_endpoint(
                    client, agent_name, foundation_model, system_prompt, vector_indexes
                )
                
                if endpoint_url:
                    st.success(f"🎉 Databricks Agent '{agent_name}' created successfully!")
                    st.info("⏳ The endpoint is being deployed. This may take 5-10 minutes.")
                    
                    # Show next steps
                    st.markdown("**Next Steps:**")
                    st.markdown("1. ✅ Agent endpoint created")
                    st.markdown("2. ⏳ Deployment in progress...")
                    st.markdown("3. 🔄 Check status above to monitor deployment")
                    st.markdown("4. 💬 Test the agent once deployed")
                    
                    # Auto-refresh status
                    st.rerun()
                    
                else:
                    st.error("❌ Failed to create agent. Check the messages above for details.")
                    
            except Exception as e:
                st.error(f"❌ Agent creation failed: {str(e)}")

def _render_external_agent_form(client: WorkspaceClient):
    """Render external model agent creation form"""
    st.subheader("🌐 External Model Agent")
    
    # Vector index selection
    st.markdown("**Knowledge Base Configuration (Optional):**")
    vector_indexes = _render_vector_index_selection(client, "ext")
    
    # Main form
    with st.form("external_agent_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        
        with col1:
            agent_name = st.text_input(
                "Agent Name *",
                value="my_external_agent",
                help="Unique name for your agent (no spaces, use underscores)",
                placeholder="e.g., openai_assistant"
            )
            
            model_provider = st.selectbox(
                "Model Provider *",
                options=["OpenAI", "Anthropic", "Google", "Cohere"],
                help="Choose your external model provider"
            )
            
            # Model selection based on provider
            if model_provider == "OpenAI":
                available_models = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"]
            elif model_provider == "Anthropic":
                available_models = ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
            elif model_provider == "Google":
                available_models = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"]
            else:  # Cohere
                available_models = ["command-r-plus", "command-r", "command", "command-light"]
            
            selected_model = st.selectbox(
                "Model *",
                options=available_models,
                help=f"Select a {model_provider} model"
            )
        
        with col2:
            api_key = st.text_input(
                f"{model_provider} API Key *",
                type="password",
                help=f"Enter your {model_provider} API key",
                placeholder="sk-... or your API key"
            )
            
            agent_description = st.text_area(
                "Agent Description",
                placeholder="Describe what your agent does and its purpose...",
                help="This helps you identify the agent later"
            )
            
            max_tokens = st.slider(
                "Max Response Tokens",
                min_value=100,
                max_value=4000,
                value=1000,
                help="Maximum length of agent responses"
            )
        
        system_prompt = st.text_area(
            "System Prompt *",
            value="""You are a helpful AI assistant. Answer questions accurately and concisely. If you don't know the answer, say so clearly. Be professional and helpful in your responses.""",
            height=120,
            help="Define how your agent should behave and respond to users",
            max_chars=2000
        )
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            temperature = st.slider(
                "Temperature (Creativity)",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Higher values make responses more creative but less predictable"
            )
            
            enable_streaming = st.checkbox(
                "Enable Response Streaming",
                value=True,
                help="Stream responses word by word for better user experience"
            )
            
            # Provider-specific settings
            if model_provider == "OpenAI":
                top_p = st.slider("Top P", 0.0, 1.0, 1.0, 0.1, help="Nucleus sampling parameter")
            elif model_provider == "Anthropic":
                top_k = st.slider("Top K", 1, 100, 40, 1, help="Top-k sampling parameter")
        
        create_external_agent = st.form_submit_button(
            f"🌐 Create {model_provider} Agent",
            use_container_width=True,
            type="primary"
        )
    
    if create_external_agent:
        if not _validate_external_agent_form(agent_name, api_key, system_prompt, model_provider):
            return
        
        clear_current_step_messages()
        
        with st.spinner(f"Creating {model_provider} agent endpoint..."):
            try:
                endpoint_url = create_agent_endpoint(
                    client, agent_name, selected_model, system_prompt, vector_indexes, api_key
                )
                
                if endpoint_url:
                    st.success(f"🎉 {model_provider} Agent '{agent_name}' created successfully!")
                    st.info("⏳ The endpoint is being deployed. This may take 5-10 minutes.")
                    
                    # Show next steps
                    st.markdown("**Next Steps:**")
                    st.markdown("1. ✅ Agent endpoint created")
                    st.markdown("2. ⏳ Deployment in progress...")
                    st.markdown("3. 🔄 Check status above to monitor deployment")
                    st.markdown("4. 💬 Test the agent once deployed")
                    
                    # Security reminder
                    st.warning("🔒 **Security Note:** Your API key is stored securely in the Databricks serving endpoint.")
                    
                    # Auto-refresh status
                    st.rerun()
                    
                else:
                    st.error("❌ Failed to create agent. Check the messages above for details.")
                    
            except Exception as e:
                st.error(f"❌ Agent creation failed: {str(e)}")

def _render_vector_index_selection(client: WorkspaceClient, prefix: str) -> list:
    """Render vector index selection UI"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        available_catalogs = get_catalogs(client)
        vector_catalog = st.selectbox(
            "Vector Catalog", 
            options=["None"] + available_catalogs,
            key=f"{prefix}_vector_catalog",
            help="Select catalog containing your vector indexes"
        )
    
    with col2:
        if vector_catalog and vector_catalog != "None":
            available_schemas = get_schemas(client, vector_catalog)
            vector_schema = st.selectbox(
                "Vector Schema", 
                options=available_schemas,
                key=f"{prefix}_vector_schema",
                help="Select schema containing your vector indexes"
            )
        else:
            vector_schema = None
            st.selectbox(
                "Vector Schema", 
                options=[], 
                disabled=True,
                key=f"{prefix}_vector_schema_disabled",
                help="Select a catalog first"
            )
    
    with col3:
        selected_vector_indexes = []
        if vector_catalog and vector_catalog != "None" and vector_schema:
            try:
                available_indexes = list_vector_search_indexes(client, vector_catalog, vector_schema)
                
                if available_indexes:
                    index_names = [idx['name'] for idx in available_indexes]
                    selected_vector_indexes = st.multiselect(
                        "Vector Indexes",
                        options=index_names,
                        key=f"{prefix}_vector_indexes",
                        help="Select vector indexes for RAG capabilities"
                    )
                    
                    if selected_vector_indexes:
                        st.success(f"✅ Selected {len(selected_vector_indexes)} index(es)")
                        
                        # Show index details
                        with st.expander("📊 Selected Index Details"):
                            for idx_name in selected_vector_indexes:
                                idx_info = next((idx for idx in available_indexes if idx['name'] == idx_name), None)
                                if idx_info:
                                    st.write(f"**{idx_name}:**")
                                    st.write(f"- Full name: `{idx_info['full_name']}`")
                                    st.write(f"- Type: {idx_info['table_type']}")
                                    if idx_info['comment']:
                                        st.write(f"- Description: {idx_info['comment']}")
                else:
                    st.multiselect(
                        "Vector Indexes", 
                        options=[], 
                        disabled=True,
                        key=f"{prefix}_vector_indexes_disabled",
                        help="No vector indexes found"
                    )
                    st.info(f"No vector indexes found in {vector_catalog}.{vector_schema}")
            except Exception as e:
                st.multiselect(
                    "Vector Indexes", 
                    options=[], 
                    disabled=True,
                    key=f"{prefix}_vector_indexes_error",
                    help="Error loading indexes"
                )
                st.error(f"Error loading indexes: {str(e)}")
        else:
            st.multiselect(
                "Vector Indexes", 
                options=[], 
                disabled=True,
                key=f"{prefix}_vector_indexes_none",
                help="Select catalog and schema first"
            )
    
    # Convert to full names for the actual creation
    full_index_names = []
    if selected_vector_indexes and vector_catalog and vector_schema:
        for idx_name in selected_vector_indexes:
            full_index_names.append(f"{vector_catalog}.{vector_schema}.{idx_name}")
    
    return full_index_names

def _validate_agent_form(agent_name: str, foundation_model: str, system_prompt: str) -> bool:
    """Validate Databricks agent form"""
    if not agent_name or not foundation_model or not system_prompt:
        st.error("❌ Please fill in all required fields (marked with *)")
        return False
    
    if " " in agent_name:
        st.error("❌ Agent name cannot contain spaces. Use underscores instead.")
        return False
    
    if len(agent_name) < 3:
        st.error("❌ Agent name must be at least 3 characters long.")
        return False
    
    if len(system_prompt.strip()) < 10:
        st.error("❌ System prompt must be at least 10 characters long.")
        return False
    
    return True

def _validate_external_agent_form(agent_name: str, api_key: str, system_prompt: str, provider: str) -> bool:
    """Validate external agent form"""
    if not agent_name or not api_key or not system_prompt:
        st.error("❌ Please fill in all required fields (marked with *)")
        return False
    
    if " " in agent_name:
        st.error("❌ Agent name cannot contain spaces. Use underscores instead.")
        return False
    
    if len(agent_name) < 3:
        st.error("❌ Agent name must be at least 3 characters long.")
        return False
    
    if len(system_prompt.strip()) < 10:
        st.error("❌ System prompt must be at least 10 characters long.")
        return False
    
    # Validate API key format
    if not _validate_api_key(provider, api_key):
        st.error(f"❌ Invalid {provider} API key format")
        return False
    
    return True

def _validate_api_key(provider: str, api_key: str) -> bool:
    """Validate API key format"""
    if not api_key or len(api_key.strip()) < 10:
        return False
    
    if provider == "OpenAI":
        return api_key.startswith("sk-") or api_key.startswith("org-")
    elif provider == "Anthropic":
        return api_key.startswith("sk-ant-") or len(api_key) > 20
    elif provider == "Google":
        return len(api_key) > 20  # Google API keys are typically long
    elif provider == "Cohere":
        return len(api_key) > 20  # Cohere API keys are typically long
    
    return True  # Default to true for unknown providers

# Sidebar help and management functions
def show_agent_creation_help():
    """Show help information for agent creation"""
    st.sidebar.markdown("### 🤖 Agent Creation Help")
    
    with st.sidebar.expander("📖 Getting Started"):
        st.markdown("""
        **Steps to create an agent:**
        
        1. **Choose Agent Type**: Select between Databricks or external models
        2. **Configure Knowledge Base**: Optionally add vector indexes for RAG
        3. **Set System Prompt**: Define your agent's behavior
        4. **Create & Deploy**: Click create and wait for deployment
        5. **Test & Use**: Test your agent and integrate it into applications
        """)
    
    with st.sidebar.expander("🔧 Best Practices"):
        st.markdown("""
        **Agent Naming:**
        - Use descriptive names (e.g., `customer_support_agent`)
        - No spaces - use underscores
        - Keep it short but meaningful
        
        **System Prompts:**
        - Be specific about the agent's role
        - Include response format guidelines
        - Mention knowledge base usage if applicable
        - Keep it concise but comprehensive
        
        **Vector Indexes:**
        - Only select relevant indexes for your use case
        - Ensure indexes are properly configured
        - Test RAG performance with sample queries
        """)
    
    with st.sidebar.expander("⚠️ Troubleshooting"):
        st.markdown("""
        **Common Issues:**
        
        - **Model not available**: Try different models from the list
        - **Permission denied**: Contact workspace admin for endpoint creation rights
        - **Deployment timeout**: Large models need 10+ minutes to deploy
        - **API key errors**: Verify key format and validity
        - **Vector index errors**: Ensure indexes exist and are accessible
        
        **Getting Help:**
        - Check the status messages for detailed error info
        - Verify your Databricks permissions
        - Contact your admin for workspace-specific issues
        """)

def render_agent_management_section():
    """Render agent management section"""
    if hasattr(st.session_state, 'custom_endpoints') and st.session_state.custom_endpoints:
        st.subheader("🔧 Agent Management")
        
        # Bulk actions
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Refresh All Status", use_container_width=True):
                _refresh_all_endpoint_status()
        
        with col2:
            if st.button("📊 Export Configuration", use_container_width=True):
                _export_agent_configurations()
        
        with col3:
            if st.button("🗑️ Clear All Records", use_container_width=True, type="secondary"):
                if st.session_state.get('confirm_clear', False):
                    st.session_state.custom_endpoints = []
                    st.success("✅ All agent records cleared")
                    st.session_state.confirm_clear = False
                    st.rerun()
                else:
                    st.session_state.confirm_clear = True
                    st.warning("⚠️ Click again to confirm deletion of all records")

def _refresh_all_endpoint_status():
    """Refresh status for all endpoints"""
    if not hasattr(st.session_state, 'custom_endpoints'):
        return
    
    try:
        client = WorkspaceClient(
            host=st.session_state.get('workspace_url', ''),
            token=st.session_state.get('databricks_token', '')
        )
        
        updated_count = 0
        for endpoint in st.session_state.custom_endpoints:
            try:
                status_info = check_endpoint_status(client, endpoint['endpoint_name'])
                endpoint['status'] = status_info.get('status', 'Unknown')
                updated_count += 1
            except Exception as e:
                endpoint['status'] = 'Error'
        
        st.success(f"✅ Updated status for {updated_count} endpoints")
        st.rerun()
    except Exception as e:
        st.error(f"❌ Error refreshing statuses: {str(e)}")

def _export_agent_configurations():
    """Export agent configurations as JSON"""
    if not hasattr(st.session_state, 'custom_endpoints'):
        st.warning("No agents to export")
        return
    
    import json
    from datetime import datetime
    
    # Prepare export data without sensitive information
    export_data = {
        'exported_at': datetime.now().isoformat(),
        'export_version': '1.0',
        'agents': []
    }
    
    for agent in st.session_state.custom_endpoints:
        clean_agent = {
            'name': agent['name'],
            'endpoint_name': agent['endpoint_name'],
            'model': agent['model'],
            'system_prompt': agent['system_prompt'],
            'vector_indexes': agent.get('vector_indexes', []),
            'is_external': agent.get('is_external', False),
            'status': agent['status'],
            'created_at': agent.get('created_at', 0)
        }
        # Note: API keys are intentionally excluded for security
        export_data['agents'].append(clean_agent)
    
    json_str = json.dumps(export_data, indent=2)
    
    st.download_button(
        label="💾 Download Configuration",
        data=json_str,
        file_name=f"agent_configurations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        help="Downloads agent configurations (API keys excluded for security)"
    )