import streamlit as st
import time
from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
from utils import (
    get_catalogs, get_schemas, get_serving_endpoints,
    chat_with_endpoint, clear_current_step_messages, add_status_message
)
import requests
import io
from contextlib import redirect_stdout

def get_tables_from_catalog_schema(client: WorkspaceClient, catalog_name: str, schema_name: str):
    """Get all tables from a specific catalog and schema"""
    try:
        tables = list(client.tables.list(catalog_name=catalog_name, schema_name=schema_name))
        table_info = []
        
        for table in tables:
            table_info.append({
                'name': table.name,
                'full_name': table.full_name,
                'table_type': table.table_type.value if table.table_type else 'Unknown',
                'comment': table.comment or 'No description'
            })
        
        return table_info
    except Exception as e:
        add_status_message(f"❌ Error fetching tables: {str(e)}", False)
        return []

def get_vector_search_indexes(client: WorkspaceClient, catalog_name: str, schema_name: str):
    """Get vector search indexes from a specific catalog and schema - FIXED VERSION"""
    try:
        # Initialize vector search client
        vs_client = VectorSearchClient(workspace_url=client.config.host, personal_access_token=client.config.token)
        
        # Get all vector search endpoints
        endpoints = vs_client.list_endpoints()
        
        vector_indexes = []
        for endpoint in endpoints.get('endpoints', []):
            if endpoint.get('endpoint_status', {}).get('state') == 'ONLINE':
                try:
                    # FIXED: Use correct API method - list_indexes without endpoint_name parameter
                    # Instead, we'll use direct API call or get all indexes and filter
                    endpoint_name = endpoint['name']
                    
                    # Try to get indexes using direct API call
                    try:
                        response = requests.get(
                            f"{client.config.host}/api/2.0/vector-search/indexes",
                            headers={"Authorization": f"Bearer {client.config.token}"},
                            params={"endpoint_name": endpoint_name}
                        )
                        
                        if response.status_code == 200:
                            indexes_data = response.json()
                            
                            for index in indexes_data.get('vector_indexes', []):
                                index_name = index.get('name', '')
                                if index_name.startswith(f"{catalog_name}.{schema_name}."):
                                    vector_indexes.append({
                                        'name': index_name.split('.')[-1],  # Just the index name
                                        'full_name': index_name,
                                        'endpoint': endpoint_name,
                                        'status': index.get('status', {}).get('ready', False)
                                    })
                        else:
                            print(f"Failed to get indexes for endpoint {endpoint_name}: {response.text}")
                            
                    except Exception as api_error:
                        print(f"API call failed for endpoint {endpoint_name}: {api_error}")
                        
                        # Fallback: try the SDK method without endpoint_name
                        try:
                            all_indexes = vs_client.list_indexes()
                            for index in all_indexes.get('vector_indexes', []):
                                index_name = index.get('name', '')
                                if (index_name.startswith(f"{catalog_name}.{schema_name}.") and 
                                    index.get('endpoint_name') == endpoint_name):
                                    vector_indexes.append({
                                        'name': index_name.split('.')[-1],
                                        'full_name': index_name,
                                        'endpoint': endpoint_name,
                                        'status': index.get('status', {}).get('ready', False)
                                    })
                        except Exception as fallback_error:
                            print(f"Fallback method also failed: {fallback_error}")
                            
                except Exception as e:
                    st.warning(f"Could not list indexes for endpoint {endpoint['name']}: {str(e)}")
                    continue
        
        return vector_indexes
    except Exception as e:
        add_status_message(f"❌ Error fetching vector indexes: {str(e)}", False)
        return []

def perform_vector_search(workspace_url: str, token: str, index_name: str, query: str, num_results: int = 5):
    """FIXED: Perform vector search with correct API usage"""
    try:
        print(f"🔍 Starting vector search on index: {index_name}")
        print(f"📝 Query: {query}")
        
        # Initialize vector search client
        try:
            vs_client = VectorSearchClient(workspace_url=workspace_url, personal_access_token=token)
            print("✅ Vector search client initialized")
        except Exception as client_error:
            print(f"❌ Failed to initialize vector search client: {str(client_error)}")
            add_status_message(f"❌ Failed to initialize vector search client: {str(client_error)}", False)
            return []
        
        # Get the vector search index
        try:
            vector_index = vs_client.get_index(index_name=index_name)
            print(f"✅ Retrieved vector index: {index_name}")
        except Exception as index_error:
            print(f"❌ Failed to get vector index {index_name}: {str(index_error)}")
            add_status_message(f"❌ Vector index {index_name} not accessible: {str(index_error)}", False)
            return []
        
        # FIXED: Perform similarity search with required columns parameter
        try:
            print(f"🔎 Performing similarity search...")
            
            # Based on your debug report, the 'content' column works
            # Try the most likely column combinations first
            column_combinations = [
                ["content"],  # This works based on your debug
                ["text"],
                ["content", "source_file"],
                ["content", "path"],
                ["text", "source"],
                ["content", "source_file", "chunk_index"]
            ]
            
            results = None
            successful_columns = None
            
            for columns in column_combinations:
                try:
                    print(f"🔍 Trying columns: {columns}")
                    results = vector_index.similarity_search(
                        query_text=query,
                        columns=columns,  # This is required according to your debug report
                        num_results=num_results
                    )
                    successful_columns = columns
                    print(f"✅ Search successful with columns: {columns}")
                    break
                except Exception as col_error:
                    print(f"❌ Failed with columns {columns}: {str(col_error)}")
                    continue
            
            if results is None:
                print("❌ All column combinations failed")
                add_status_message("❌ Vector search failed with all column combinations", False)
                return []
            
        except Exception as search_error:
            print(f"❌ Similarity search failed: {str(search_error)}")
            add_status_message(f"❌ Similarity search failed: {str(search_error)}", False)
            return []
        
        # Extract relevant context with improved parsing
        context_chunks = []
        try:
            print(f"📊 Processing search results: {type(results)}")
            
            if results:
                # Handle different result formats more robustly
                data_array = None
                
                # Check for different result structures
                if hasattr(results, 'result'):
                    if hasattr(results.result, 'data_array'):
                        data_array = results.result.data_array
                    elif hasattr(results.result, 'data'):
                        data_array = results.result.data
                elif isinstance(results, dict):
                    data_array = results.get('result', {}).get('data_array', [])
                    if not data_array:
                        data_array = results.get('data_array', [])
                        if not data_array:
                            data_array = results.get('data', [])
                elif isinstance(results, list):
                    data_array = results
                else:
                    # Try to extract any iterable data
                    try:
                        data_array = list(results) if hasattr(results, '__iter__') else []
                    except:
                        data_array = []
                
                print(f"📋 Data array type: {type(data_array)}")
                print(f"📋 Data array length: {len(data_array) if data_array else 0}")
                print(f"📋 Using columns: {successful_columns}")
                
                if data_array:
                    for i, item in enumerate(data_array):
                        print(f"🔍 Processing item {i}: {type(item)} - {item}")
                        
                        try:
                            text_content = None
                            
                            if isinstance(item, (list, tuple)) and len(item) > 0:
                                # Extract text content (usually first column which should be 'content')
                                text_content = str(item[0]) if item[0] is not None else ""
                                
                                # Add source information if available
                                source_info = ""
                                if len(item) > 1 and item[1] is not None:
                                    source_info = f" (Source: {item[1]})"
                                if len(item) > 2 and item[2] is not None:
                                    source_info += f" (Chunk: {item[2]})"
                                
                                if source_info:
                                    print(f"📍 Source info: {source_info}")
                                
                            elif isinstance(item, dict):
                                # Handle dictionary format
                                content_keys = ['content', 'text', 'document', 'chunk', 'data']
                                for key in content_keys:
                                    if key in item and item[key]:
                                        text_content = str(item[key])
                                        print(f"📝 Found content in key '{key}': {text_content[:100]}...")
                                        break
                                        
                            else:
                                # Try to convert directly to string
                                text_content = str(item) if item else ""
                            
                            # Validate and add content
                            if text_content and len(text_content.strip()) > 10:  # Minimum content length
                                clean_content = text_content.strip()
                                context_chunks.append(clean_content)
                                print(f"✅ Added context chunk ({len(clean_content)} chars): {clean_content[:100]}...")
                            else:
                                print(f"⚠️ Skipped item {i}: content too short or empty")
                                
                        except Exception as item_error:
                            print(f"❌ Error processing item {i}: {item_error}")
                            continue
                else:
                    print("❌ No data array found in results")
            else:
                print("❌ No results returned from similarity search")
        
        except Exception as parsing_error:
            print(f"❌ Error parsing search results: {str(parsing_error)}")
            add_status_message(f"❌ Error parsing search results: {str(parsing_error)}", False)
        
        print(f"📊 Final context chunks count: {len(context_chunks)}")
        
        if context_chunks:
            add_status_message(f"✅ Retrieved {len(context_chunks)} relevant chunks from vector search")
            return context_chunks
        else:
            add_status_message("⚠️ No relevant content found in vector search results", False)
            return []
        
    except Exception as e:
        error_msg = f"Vector search error on {index_name}: {str(e)}"
        print(f"❌ {error_msg}")
        add_status_message(f"❌ {error_msg}", False)
        return []

def enhanced_chat_with_endpoint(client: WorkspaceClient, endpoint_name: str, user_message: str, 
                               system_prompt: str, vector_indexes: list, workspace_url: str, token: str):
    """Enhanced chat function with improved vector search context"""
    try:
        print(f"🚀 Starting enhanced chat with endpoint: {endpoint_name}")
        print(f"📝 User message: {user_message}")
        print(f"📊 Vector indexes to search: {vector_indexes}")
        
        # Gather context from vector indexes
        context_chunks = []
        
        if vector_indexes:
            print(f"🔍 Processing {len(vector_indexes)} vector indexes...")
            
            for i, index_name in enumerate(vector_indexes):
                print(f"🔍 Searching index {i+1}/{len(vector_indexes)}: {index_name}")
                
                try:
                    # Ensure we have the full index name
                    if index_name.count('.') < 2:
                        print(f"⚠️ Index name '{index_name}' appears incomplete (should be catalog.schema.index)")
                        add_status_message(f"⚠️ Skipping incomplete index name: {index_name}", False)
                        continue
                    
                    chunks = perform_vector_search(workspace_url, token, index_name, user_message, num_results=3)
                    
                    if chunks:
                        context_chunks.extend(chunks)
                        print(f"✅ Added {len(chunks)} chunks from {index_name}")
                    else:
                        print(f"❌ No chunks returned from {index_name}")
                        
                except Exception as index_error:
                    error_msg = f"Error searching index {index_name}: {str(index_error)}"
                    print(f"❌ {error_msg}")
                    add_status_message(f"❌ {error_msg}", False)
                    continue
        else:
            print("ℹ️ No vector indexes provided for context search")
        
        # Build the complete prompt with context
        if context_chunks:
            print(f"📋 Building prompt with {len(context_chunks)} context chunks")
            
            # Limit and clean context
            limited_chunks = context_chunks[:5]  # Limit to top 5 chunks
            context_text = "\n\n---\n\n".join(limited_chunks)
            
            # Ensure context isn't too long (limit to ~2000 chars to leave room for other content)
            if len(context_text) > 2000:
                context_text = context_text[:2000] + "... [truncated]"
                print("✂️ Truncated context due to length")
            
            enhanced_prompt = f"""
{system_prompt}

RELEVANT CONTEXT FROM KNOWLEDGE BASE:
{context_text}

USER QUESTION: {user_message}

Please answer the user's question based on the provided context. If the answer is not present in the provided context, please say "Answer not in the provided context".
"""
            
            print("✅ Enhanced prompt created with context")
            print(f"📏 Enhanced prompt length: {len(enhanced_prompt)} characters")
            add_status_message(f"✅ Added context from {len(limited_chunks)} knowledge chunks")
            
        else:
            print("❌ No context chunks available")
            if vector_indexes:
                add_status_message("⚠️ No relevant context found in any vector indexes", False)
            
            enhanced_prompt = f"""
{system_prompt}

USER QUESTION: {user_message}

Note: No relevant context was found in the knowledge base for this question. Please provide a general answer if possible, or indicate that you need more specific information.
"""
        
        print(f"📤 Sending request to endpoint: {endpoint_name}")
        
        # Call the endpoint with the enhanced prompt
        response = chat_with_endpoint(client, endpoint_name, enhanced_prompt, "", [])
        
        print(f"📥 Received response: {response[:100]}...")
        return response
        
    except Exception as e:
        error_msg = f"Enhanced chat error: {str(e)}"
        print(f"❌ {error_msg}")
        add_status_message(f"❌ {error_msg}", False)
        return f"Sorry, I encountered an error: {error_msg}"

def debug_vector_search_setup(client: WorkspaceClient, vector_indexes: list, workspace_url: str, token: str):
    """FIXED: Debug function with correct API usage"""
    debug_output = []
    
    try:
        debug_output.append("🔍 VECTOR SEARCH DEBUG REPORT")
        debug_output.append("=" * 50)
        
        # 1. Check Vector Search Client
        try:
            vs_client = VectorSearchClient(workspace_url=workspace_url, personal_access_token=token)
            debug_output.append("✅ Vector Search Client: OK")
            
            # List all endpoints
            try:
                endpoints = vs_client.list_endpoints()
                debug_output.append(f"📊 Available Vector Search Endpoints: {len(endpoints.get('endpoints', []))}")
                
                for endpoint in endpoints.get('endpoints', []):
                    endpoint_name = endpoint.get('name', 'Unknown')
                    endpoint_state = endpoint.get('endpoint_status', {}).get('state', 'Unknown')
                    debug_output.append(f"   - {endpoint_name}: {endpoint_state}")
                    
                    # FIXED: List indexes using correct method
                    try:
                        # Try direct API call first
                        response = requests.get(
                            f"{client.config.host}/api/2.0/vector-search/indexes",
                            headers={"Authorization": f"Bearer {client.config.token}"},
                            params={"endpoint_name": endpoint_name}
                        )
                        
                        if response.status_code == 200:
                            indexes_data = response.json()
                            endpoint_indexes = indexes_data.get('vector_indexes', [])
                            debug_output.append(f"     📋 Indexes: {len(endpoint_indexes)}")
                            
                            for idx in endpoint_indexes:
                                idx_name = idx.get('name', 'Unknown')
                                idx_status = idx.get('status', {})
                                debug_output.append(f"       - {idx_name}: {idx_status}")
                        else:
                            debug_output.append(f"     ❌ API call failed: {response.status_code}")
                            
                    except Exception as idx_error:
                        debug_output.append(f"     ❌ Cannot list indexes: {idx_error}")
                        
            except Exception as endpoint_error:
                debug_output.append(f"❌ Cannot list endpoints: {endpoint_error}")
                
        except Exception as client_error:
            debug_output.append(f"❌ Vector Search Client: FAILED - {client_error}")
            return "\n".join(debug_output)
        
        # 2. Check specific indexes - FIXED
        debug_output.append(f"\n🔍 Checking specified vector indexes: {vector_indexes}")
        
        for index_name in vector_indexes:
            debug_output.append(f"\n📊 Testing index: {index_name}")
            
            try:
                vector_index = vs_client.get_index(index_name=index_name)
                debug_output.append(f"✅ Index accessible: {index_name}")
                
                # FIXED: Try search with required columns parameter
                column_tests = [["content"], ["text"], ["content", "source_file"]]
                search_successful = False
                
                for cols in column_tests:
                    try:
                        test_results = vector_index.similarity_search(
                            query_text="test query",
                            columns=cols,  # Required parameter
                            num_results=1
                        )
                        debug_output.append(f"✅ Search with columns {cols} passed")
                        debug_output.append(f"📋 Result type: {type(test_results)}")
                        
                        # Try to extract some sample data
                        if test_results:
                            try:
                                if hasattr(test_results, 'result'):
                                    data = test_results.result
                                    if hasattr(data, 'data_array') and data.data_array:
                                        debug_output.append(f"📋 Sample result: {data.data_array[0] if data.data_array else 'No data'}")
                                elif isinstance(test_results, dict):
                                    debug_output.append(f"📋 Dict keys: {list(test_results.keys())}")
                            except Exception as sample_error:
                                debug_output.append(f"⚠️ Could not extract sample: {sample_error}")
                        
                        search_successful = True
                        break
                        
                    except Exception as col_error:
                        debug_output.append(f"❌ Search with columns {cols} failed: {col_error}")
                
                if not search_successful:
                    debug_output.append("❌ All search methods failed")
                    
            except Exception as index_error:
                debug_output.append(f"❌ Index not accessible: {index_error}")
        
        # 3. Check session state
        debug_output.append(f"\n🔍 Session State Check:")
        if hasattr(st.session_state, 'vectorized_indexes'):
            debug_output.append(f"📊 Vectorized indexes in session: {len(st.session_state.vectorized_indexes)}")
            for idx in st.session_state.vectorized_indexes:
                debug_output.append(f"   - {idx.get('name', 'Unknown')}: {idx.get('index_path', 'No path')}")
        else:
            debug_output.append("❌ No vectorized_indexes in session state")
        
        if hasattr(st.session_state, 'agent_endpoints'):
            debug_output.append(f"📊 Agent endpoints in session: {len(st.session_state.agent_endpoints)}")
            for agent in st.session_state.agent_endpoints:
                debug_output.append(f"   - {agent.get('name', 'Unknown')}: {agent.get('vector_indexes', [])}")
        else:
            debug_output.append("❌ No agent_endpoints in session state")
        
        debug_output.append("\n" + "=" * 50)
        debug_output.append("🔍 DEBUG REPORT COMPLETE")
        
    except Exception as e:
        debug_output.append(f"❌ Debug failed: {str(e)}")
    
    return "\n".join(debug_output)

def render_chat_tab(workspace_url, token):
    """Render the Chat tab"""
    st.subheader("💬 Chat with Agent")
    
    if not (workspace_url and token):
        st.warning("⚠️ Please configure Databricks connection first")
    else:
        try:
            client = WorkspaceClient(host=workspace_url, token=token)
            
            # Get all available serving endpoints
            all_endpoints = get_serving_endpoints(client)
            
            if all_endpoints:
                # Endpoint selection and system prompt
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    endpoint_options = [f"{ep['name']} ({ep['state']})" for ep in all_endpoints]
                    selected_endpoint_display = st.selectbox(
                        "Select Endpoint to Chat With",
                        options=endpoint_options,
                        help="Choose from all available serving endpoints in your workspace"
                    )
                    
                    # Extract endpoint name
                    selected_endpoint_name = selected_endpoint_display.split(' (')[0] if selected_endpoint_display else None
                
                with col2:
                    if st.button("🔄 Refresh Endpoints"):
                        st.rerun()
                
                if selected_endpoint_name:
                    st.divider()
                    
                    # System prompt input
                    system_prompt = st.text_area(
                        "System Prompt",
                        value="You are a helpful AI assistant. Answer questions based on the provided context from the knowledge base. If the answer is not present in the provided context then just say 'Answer not in the provided context' Do not provide wrong answers",
                        help="Customize the system prompt for this chat session",
                        key=f"system_prompt_{selected_endpoint_name}"
                    )
                    
                    # Vector context selection
                    st.write("**Vector Context Selection (Optional):**")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        available_catalogs = get_catalogs(client)
                        chat_vector_catalog = st.selectbox(
                            "Vector Catalog",
                            options=["None"] + available_catalogs,
                            key=f"chat_vector_catalog_{selected_endpoint_name}"
                        )
                    
                    with col2:
                        if chat_vector_catalog and chat_vector_catalog != "None":
                            available_schemas = get_schemas(client, chat_vector_catalog)
                            chat_vector_schema = st.selectbox(
                                "Vector Schema",
                                options=available_schemas,
                                key=f"chat_vector_schema_{selected_endpoint_name}"
                            )
                        else:
                            chat_vector_schema = None
                            st.selectbox(
                                "Vector Schema",
                                options=[],
                                disabled=True,
                                key=f"chat_vector_schema_disabled_{selected_endpoint_name}"
                            )
                    
                    with col3:
                        selected_vector_indexes = []
                        if chat_vector_catalog and chat_vector_catalog != "None" and chat_vector_schema:
                            # Option to select between vector indexes and regular tables
                            selection_type = st.radio(
                                "Select from:",
                                ["Vector Indexes", "All Tables"],
                                key=f"selection_type_{selected_endpoint_name}",
                                help="Choose whether to search for vector indexes specifically or show all tables"
                            )
                            
                            if selection_type == "Vector Indexes":
                                # Get vector search indexes
                                available_indexes = get_vector_search_indexes(client, chat_vector_catalog, chat_vector_schema)
                                
                                if available_indexes:
                                    index_options = []
                                    for idx in available_indexes:
                                        status_icon = "✅" if idx['status'] else "⏳"
                                        index_options.append(f"{idx['name']} {status_icon}")
                                    
                                    selected_displays = st.multiselect(
                                        "Vector Indexes",
                                        options=index_options,
                                        help="Select vector search indexes to provide context",
                                        key=f"chat_vector_indexes_{selected_endpoint_name}"
                                    )
                                    
                                    # Extract full names for selected indexes
                                    selected_vector_indexes = []
                                    for display in selected_displays:
                                        index_name = display.split(' ')[0]  # Remove status icon
                                        for idx in available_indexes:
                                            if idx['name'] == index_name:
                                                selected_vector_indexes.append(idx['full_name'])
                                                break
                                    
                                    if selected_vector_indexes:
                                        st.success(f"✅ Selected {len(selected_vector_indexes)} vector index(es)")
                                else:
                                    st.info(f"No vector indexes found in {chat_vector_catalog}.{chat_vector_schema}")
                                    st.multiselect("Vector Indexes", options=[], disabled=True)
                            
                            else:  # All Tables
                                # Fetch tables from the selected catalog and schema
                                available_tables = get_tables_from_catalog_schema(client, chat_vector_catalog, chat_vector_schema)
                                
                                if available_tables:
                                    table_options = [f"{table['name']} ({table['table_type']})" for table in available_tables]
                                    
                                    selected_table_displays = st.multiselect(
                                        "Tables",
                                        options=table_options,
                                        help="Select tables to provide context for the conversation",
                                        key=f"chat_tables_{selected_endpoint_name}"
                                    )
                                    
                                    # Extract table names and construct full names
                                    selected_vector_indexes = []
                                    for display in selected_table_displays:
                                        table_name = display.split(' (')[0]
                                        full_name = f"{chat_vector_catalog}.{chat_vector_schema}.{table_name}"
                                        selected_vector_indexes.append(full_name)
                                    
                                    if selected_vector_indexes:
                                        st.success(f"✅ Selected {len(selected_vector_indexes)} table(s)")
                                        
                                        # Show details of selected tables
                                        with st.expander("📋 Selected Table Details"):
                                            for table in available_tables:
                                                table_full_name = f"{chat_vector_catalog}.{chat_vector_schema}.{table['name']}"
                                                if table_full_name in selected_vector_indexes:
                                                    st.write(f"**{table['name']}** ({table['table_type']})")
                                                    st.write(f"Full name: `{table['full_name']}`")
                                                    if table['comment'] != 'No description':
                                                        st.write(f"Description: {table['comment']}")
                                                    st.write("---")
                                else:
                                    st.info(f"No tables found in {chat_vector_catalog}.{chat_vector_schema}")
                                    st.multiselect("Tables", options=[], disabled=True)
                        else:
                            st.multiselect(
                                "Vector Indexes/Tables",
                                options=[],
                                disabled=True,
                                key=f"chat_vector_disabled_{selected_endpoint_name}"
                            )
                    
                    # Debug button - add this for troubleshooting
                    if st.button("🔍 Debug Vector Search Setup", key=f"debug_{selected_endpoint_name}"):
                        with st.expander("🔍 Debug Report", expanded=True):
                            debug_report = debug_vector_search_setup(client, selected_vector_indexes, workspace_url, token)
                            st.code(debug_report, language="text")
                    
                    st.divider()
                    
                    # Chat interface
                    st.write(f"**Chatting with:** `{selected_endpoint_name}`")
                    if selected_vector_indexes:
                        st.write(f"**Context sources:** {len(selected_vector_indexes)} selected")
                        with st.expander("📋 Selected Vector Sources"):
                            for idx in selected_vector_indexes:
                                st.write(f"- {idx}")
                    
                    # Initialize chat history for this endpoint
                    chat_key = f"chat_history_{selected_endpoint_name}"
                    if chat_key not in st.session_state:
                        st.session_state[chat_key] = []
                    
                    # Display chat history
                    chat_container = st.container()
                    with chat_container:
                        for message in st.session_state[chat_key]:
                            with st.chat_message(message["role"]):
                                st.write(message["content"])
                    
                    # Chat input
                    user_message = st.chat_input("Type your message here...")
                    
                    if user_message:
                        # Add user message to history
                        st.session_state[chat_key].append({"role": "user", "content": user_message})
                        
                        # Display user message
                        with st.chat_message("user"):
                            st.write(user_message)
                        
                        # Get response from endpoint
                        with st.chat_message("assistant"):
                            with st.spinner("Thinking..."):
                                try:
                                    # Show status messages during processing
                                    status_placeholder = st.empty()
                                    
                                    response = enhanced_chat_with_endpoint(
                                        client,
                                        selected_endpoint_name,
                                        user_message,
                                        system_prompt,
                                        selected_vector_indexes,
                                        workspace_url,
                                        token
                                    )
                                    
                                    # Clear status and show response
                                    status_placeholder.empty()
                                    st.write(response)
                                    
                                    # Add assistant response to history
                                    st.session_state[chat_key].append({"role": "assistant", "content": response})
                                    
                                    # Show any status messages
                                    if hasattr(st.session_state, 'current_step_messages') and st.session_state.current_step_messages:
                                        with st.expander("📋 Processing Details"):
                                            for msg in st.session_state.current_step_messages:
                                                if msg['is_success']:
                                                    st.success(msg['message'])
                                                else:
                                                    st.error(msg['message'])
                                        clear_current_step_messages()
                                    
                                except Exception as e:
                                    error_msg = f"Error communicating with endpoint: {str(e)}"
                                    st.error(error_msg)
                                    st.session_state[chat_key].append({"role": "assistant", "content": error_msg})
                    
                    # Chat controls
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("🗑️ Clear Chat History"):
                            st.session_state[chat_key] = []
                            st.rerun()
                    
                    with col2:
                        if st.button("📋 Export Chat"):
                            chat_export = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state[chat_key]])
                            st.download_button(
                                "💾 Download Chat",
                                data=chat_export,
                                file_name=f"chat_{selected_endpoint_name}_{int(time.time())}.txt",
                                mime="text/plain"
                            )
                    
                    with col3:
                        if st.button("🔍 Check Status"):
                            clear_current_step_messages()
                            for ep in all_endpoints:
                                if ep['name'] == selected_endpoint_name:
                                    add_status_message(f"Endpoint: {ep['name']} - Status: {ep['state']}")
                                    break
                
                # Display created agents
                if hasattr(st.session_state, 'agent_endpoints') and st.session_state.agent_endpoints:
                    st.divider()
                    st.write("**Your Created Agents:**")
                    
                    for agent in st.session_state.agent_endpoints:
                        with st.expander(f"🤖 {agent['name']}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Model:** {agent['model']}")
                                st.write(f"**Vector Indexes:** {', '.join(agent.get('vector_indexes', ['None']))}")
                                st.write(f"**Status:** {agent.get('status', 'Unknown')}")
                            
                            with col2:
                                st.code(agent['url'], language="text")
                                
                                if st.button(f"💬 Quick Chat", key=f"quick_chat_{agent['name']}"):
                                    st.info(f"Select '{agent['endpoint_name']}' from the dropdown above to start chatting!")
            
            else:
                st.info("🤖 No serving endpoints found. Create an agent first or check your endpoint permissions.")
                
                if st.button("🔄 Refresh Endpoints"):
                    st.rerun()
        
        except Exception as e:
            st.error(f"Error loading endpoints: {str(e)}")
            
            # Add troubleshooting section
            with st.expander("🔧 Troubleshooting"):
                st.write("**Common issues and solutions:**")
                st.write("1. **Token Permissions**: Ensure your token has access to serving endpoints and vector search")
                st.write("2. **Workspace URL**: Verify the workspace URL is correct (should start with https://)")
                st.write("3. **Network**: Check if you can access Databricks workspace from your network")
                st.write("4. **Endpoints**: Verify that serving endpoints exist in your workspace")
                
                if st.button("🧪 Test Connection"):
                    try:
                        test_client = WorkspaceClient(host=workspace_url, token=token)
                        catalogs = get_catalogs(test_client)
                        st.success(f"✅ Connection successful! Found {len(catalogs)} catalogs.")
                    except Exception as test_e:
                        st.error(f"❌ Connection test failed: {str(test_e)}")
                        
    # Initialize session state for status messages if not exists
    if 'current_step_messages' not in st.session_state:
        st.session_state.current_step_messages = []