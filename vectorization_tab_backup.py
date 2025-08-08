import streamlit as st
from databricks.sdk import WorkspaceClient
from utils import (
    get_catalogs, get_schemas, clear_current_step_messages,
    create_vector_search_endpoint, create_vector_search_index,
    add_status_message, get_warehouse_id
)
import requests
import time

def get_volumes_from_catalog_schema(client: WorkspaceClient, catalog_name: str, schema_name: str):
    """Get all volumes from a specific catalog and schema"""
    try:
        volumes = list(client.volumes.list(catalog_name=catalog_name, schema_name=schema_name))
        return [volume.name for volume in volumes]
    except Exception as e:
        add_status_message(f"❌ Error fetching volumes: {str(e)}", False)
        return []

def get_files_from_volume(client: WorkspaceClient, catalog_name: str, schema_name: str, volume_name: str):
    """Get all files from a specific volume"""
    try:
        volume_path = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/"
        
        response = client.files.list_directory_contents(directory_path=volume_path)
        
        files = []
        for file_info in response:
            if not file_info.is_directory:
                file_name = file_info.name
                file_extension = file_name.split('.')[-1].lower() if '.' in file_name else 'unknown'
                
                file_type = 'Unknown'
                if file_extension in ['pdf']:
                    file_type = 'PDF'
                elif file_extension in ['txt', 'md']:
                    file_type = 'Text'
                elif file_extension in ['doc', 'docx']:
                    file_type = 'Document'
                elif file_extension in ['csv']:
                    file_type = 'CSV'
                elif file_extension in ['json']:
                    file_type = 'JSON'
                
                files.append({
                    'name': file_name,
                    'path': file_info.path,
                    'size': file_info.file_size,
                    'type': file_type,
                    'extension': file_extension,
                    'volume': volume_name
                })
        
        return files
    except Exception as e:
        add_status_message(f"❌ Error fetching files from volume {volume_name}: {str(e)}", False)
        return []

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

def get_table_columns(client: WorkspaceClient, catalog_name: str, schema_name: str, table_name: str):
    """Get column information for a table"""
    try:
        full_table_name = f"{catalog_name}.{schema_name}.{table_name}"
        warehouse_id = get_warehouse_id(client)
        
        # Use DESCRIBE TABLE to get column information
        describe_sql = f"DESCRIBE TABLE {full_table_name}"
        
        result = client.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            statement=describe_sql,
            wait_timeout="30s"
        )
        
        columns = []
        if result.result and result.result.data_array:
            for row in result.result.data_array:
                if row and len(row) >= 2:
                    col_name = row[0]
                    col_type = row[1]
                    # Skip partition information and comments
                    if col_name and not col_name.startswith('#') and col_name != '':
                        columns.append({
                            'name': col_name,
                            'type': col_type
                        })
        
        return columns
    except Exception as e:
        add_status_message(f"❌ Error getting table columns: {str(e)}", False)
        return []

def get_vector_search_endpoints(client: WorkspaceClient):
    """Get list of available vector search endpoints"""
    try:
        response = requests.get(
            f"{client.config.host}/api/2.0/vector-search/endpoints",
            headers={"Authorization": f"Bearer {client.config.token}"},
            timeout=10
        )
        
        if response.status_code == 200:
            endpoints_data = response.json()
            endpoints = []
            for endpoint in endpoints_data.get("vector_indexes", []):
                endpoints.append(endpoint.get("name", ""))
            
            # Also try a different API structure
            if not endpoints and "endpoints" in endpoints_data:
                for endpoint in endpoints_data.get("endpoints", []):
                    endpoints.append(endpoint.get("name", ""))
            
            return [ep for ep in endpoints if ep]  # Filter out empty names
        else:
            add_status_message(f"⚠️ Could not fetch vector search endpoints: {response.status_code}")
            return []
    except Exception as e:
        add_status_message(f"⚠️ Error fetching vector search endpoints: {str(e)}")
        return []

def get_embedding_models(client: WorkspaceClient):
    """Get list of available embedding models"""
    try:
        # Try to get serving endpoints that might be embedding models
        response = requests.get(
            f"{client.config.host}/api/2.0/serving-endpoints",
            headers={"Authorization": f"Bearer {client.config.token}"},
            timeout=10
        )
        
        embedding_models = []
        if response.status_code == 200:
            endpoints_data = response.json()
            for endpoint in endpoints_data.get("endpoints", []):
                endpoint_name = endpoint.get("name", "")
                # Look for common embedding model patterns
                if any(pattern in endpoint_name.lower() for pattern in [
                    "bge", "embedding", "e5", "sentence", "embed"
                ]):
                    embedding_models.append(endpoint_name)
        
        # Add common Databricks embedding models as fallback
        default_models = [
            "databricks-bge-large-en",
            "databricks-gte-large-en", 
            "text-embedding-ada-002"
        ]
        
        # Combine and remove duplicates
        all_models = list(set(embedding_models + default_models))
        return sorted(all_models)
        
    except Exception as e:
        add_status_message(f"⚠️ Error fetching embedding models: {str(e)}")
        return ["databricks-bge-large-en", "databricks-gte-large-en"]

def create_text_table_from_volume_files(client: WorkspaceClient, selected_files, catalog_name: str, schema_name: str, progress_bar, status_container):
    """Create text tables from volume files using SQL parsing"""
    
    sql_template = """
    SET STATEMENT_TIMEOUT = 86400;
    DECLARE parse_extensions ARRAY<STRING> DEFAULT array('.pdf', '.jpg', '.jpeg', '.png');
    CREATE OR REPLACE TABLE {destination_table} AS (
    -- Parse documents with ai_parse
    WITH all_files AS (
      SELECT
        path,
        content
      FROM
        READ_FILES('{source_path}', format => 'binaryFile')
      WHERE path = '{specific_file_path}'
      ORDER BY
        path ASC
      LIMIT 1000
    ),
    repartitioned_files AS (
      SELECT *
      FROM all_files
      DISTRIBUTE BY crc32(path) % 4
    ),
    -- Parse the files using ai_parse document  
    parsed_documents AS (
      SELECT
        path,
        ai_parse_document(content) as parsed
      FROM
        repartitioned_files
      WHERE array_contains(parse_extensions, lower(regexp_extract(path, r'(\\.[^.]+)$', 1)))
    ),
    raw_documents AS (
      SELECT
        path,
        null as raw_parsed,
        decode(content, "utf-8") as text
      FROM 
        repartitioned_files
      WHERE NOT array_contains(parse_extensions, lower(regexp_extract(path, r'(\\.[^.]+)$', 1)))
    ),
    -- Extract page markdowns from ai_parse output
    sorted_page_contents AS (
      SELECT
        path,
        page:content AS content
      FROM
        (
          SELECT
            path,
            posexplode(try_cast(parsed:document:pages AS ARRAY<VARIANT>)) AS (page_idx, page)
          FROM
            parsed_documents
          WHERE
            parsed:document:pages IS NOT NULL
            AND CAST(parsed:error_status AS STRING) IS NULL
        )
      ORDER BY
        page_idx
    ),
    -- Concatenate so we have 1 row per document
    concatenated AS (
        SELECT
            path,
            concat_ws('\\n\\n', collect_list(content)) AS full_content
        FROM
            sorted_page_contents
        GROUP BY
            path
    ),
    -- Bring back the raw parsing since it could be useful for other downstream uses
    with_raw AS (
        SELECT
            a.path,
            b.parsed as raw_parsed,
            a.full_content as text
        FROM concatenated a
        JOIN parsed_documents b ON a.path = b.path
    )
    -- Recombine raw text documents with parsed documents
    SELECT *  FROM with_raw
    UNION ALL 
    SELECT * FROM raw_documents
    );
    """
    
    warehouse_id = get_warehouse_id(client)
    successful_tables = []
    total_files = len(selected_files)
    
    for idx, file_info in enumerate(selected_files):
        try:
            progress = (idx + 1) / total_files
            progress_bar.progress(progress)
            status_container.text(f"Processing {file_info['name']} ({idx + 1}/{total_files})")
            
            # Generate table name
            safe_name = file_info['name'].replace('.', '_').replace(' ', '_').replace('-', '_').lower()
            table_name = f"raw_text_parsed_{safe_name}"
            full_table_name = f"{catalog_name}.{schema_name}.{table_name}"
            
            # Prepare volume path
            volume_path = f"/Volumes/{catalog_name}/{schema_name}/{file_info['volume']}/*"
            
            # Create SQL with specific file path
            sql_command = sql_template.format(
                destination_table=full_table_name,
                source_path=volume_path,
                specific_file_path=file_info['path']
            )
            
            # Start async execution first
            add_status_message(f"🔄 Starting table creation for {file_info['name']}...")
            
            # Execute SQL with proper timeout and handle long operations
            try:
                # First try with wait_timeout of 0 (async)
                result = client.statement_execution.execute_statement(
                    warehouse_id=warehouse_id,
                    statement=sql_command,
                    wait_timeout="0s"  # Start async execution
                )
                
                # Get the statement_id for polling
                statement_id = result.statement_id
                add_status_message(f"📋 Statement submitted: {statement_id}")
                
                # Poll for completion
                max_polls = 30  # Max 5 minutes (30 * 10 seconds)
                poll_count = 0
                
                while poll_count < max_polls:
                    time.sleep(10)  # Wait 10 seconds between polls
                    poll_count += 1
                    
                    # Check status
                    status_result = client.statement_execution.get_statement(statement_id)
                    current_state = status_result.status.state.value
                    
                    status_container.text(f"Processing {file_info['name']} - Status: {current_state} ({poll_count}/{max_polls})")
                    
                    if current_state == "SUCCEEDED":
                        successful_tables.append({
                            'name': table_name,
                            'full_name': full_table_name,
                            'source_file': file_info['name'],
                            'source_path': file_info['path']
                        })
                        add_status_message(f"✅ Created table: {full_table_name}")
                        break
                    elif current_state in ["FAILED", "CANCELED"]:
                        error_msg = status_result.status.error.message if status_result.status.error else "Unknown error"
                        add_status_message(f"❌ Failed to create table for {file_info['name']}: {error_msg}", False)
                        break
                    elif current_state in ["RUNNING", "PENDING"]:
                        continue  # Keep polling
                    else:
                        add_status_message(f"⚠️ Unexpected status for {file_info['name']}: {current_state}")
                        break
                
                if poll_count >= max_polls:
                    add_status_message(f"⏱️ Timeout waiting for {file_info['name']} - check Databricks console", False)
                    
            except Exception as exec_error:
                # Fallback: try with shorter timeout
                try:
                    add_status_message(f"🔄 Retrying with shorter timeout for {file_info['name']}...")
                    result = client.statement_execution.execute_statement(
                        warehouse_id=warehouse_id,
                        statement=sql_command,
                        wait_timeout="50s"  # Maximum allowed synchronous timeout
                    )
                    
                    if result.status.state.value == "SUCCEEDED":
                        successful_tables.append({
                            'name': table_name,
                            'full_name': full_table_name,
                            'source_file': file_info['name'],
                            'source_path': file_info['path']
                        })
                        add_status_message(f"✅ Created table: {full_table_name}")
                    else:
                        error_msg = result.status.error.message if result.status.error else "Unknown error"
                        add_status_message(f"❌ Failed to create table for {file_info['name']}: {error_msg}", False)
                        
                except Exception as retry_error:
                    add_status_message(f"❌ Error processing {file_info['name']}: {str(retry_error)}", False)
                    continue
                
        except Exception as e:
            add_status_message(f"❌ Error processing {file_info['name']}: {str(e)}", False)
            continue
    
    progress_bar.progress(1.0)
    status_container.success(f"✅ Successfully created {len(successful_tables)}/{total_files} tables")
    
    return successful_tables

def create_vector_index(client: WorkspaceClient, table_name: str, catalog_name: str, schema_name: str,
                      primary_key: str, sync_column: str, embedding_column: str, 
                      embedding_model: str, vector_endpoint: str):
    """Create vector search index for a table"""
    try:
        # Generate index name
        index_name = f"index_{table_name}"
        full_index_name = f"{catalog_name}.{schema_name}.{index_name}"
        full_table_name = f"{catalog_name}.{schema_name}.{table_name}"
        
        # Create index configuration
        index_config = {
            "name": full_index_name,
            "endpoint_name": vector_endpoint,
            "primary_key": primary_key,
            "index_type": "DELTA_SYNC",
            "delta_sync_index_spec": {
                "source_table": full_table_name,
                "pipeline_type": "TRIGGERED",
                "embedding_source_columns": [
                    {
                        "name": embedding_column,
                        "embedding_model_endpoint_name": embedding_model
                    }
                ]
            }
        }
        
        # Add sync column if different from embedding column
        if sync_column and sync_column != embedding_column:
            index_config["delta_sync_index_spec"]["embedding_source_columns"][0]["sync_column"] = sync_column
        
        response = requests.post(
            f"{client.config.host}/api/2.0/vector-search/indexes",
            headers={
                "Authorization": f"Bearer {client.config.token}",
                "Content-Type": "application/json"
            },
            json=index_config,
            timeout=60
        )
        
        if response.status_code in [200, 201]:
            add_status_message(f"✅ Created vector index: {full_index_name}")
            return full_index_name
        else:
            error_msg = response.text
            try:
                error_json = response.json()
                error_msg = error_json.get('message', error_json.get('error_code', error_msg))
            except:
                pass
            add_status_message(f"❌ Failed to create vector index: {error_msg}", False)
            return None
            
    except Exception as e:
        add_status_message(f"❌ Error creating vector index: {str(e)}", False)
        return None

def render_vectorization_tab(workspace_url, token):
    """Render the Vectorization tab"""
    st.subheader("Data Vectorization")
    
    if not (workspace_url and token):
        st.warning("⚠️ Please configure Databricks connection first")
    else:
        try:
            client = WorkspaceClient(host=workspace_url, token=token)
            
            # Catalog and Schema selection
            col1, col2 = st.columns(2)
            with col1:
                available_catalogs = get_catalogs(client)
                selected_catalog = st.selectbox("Select Catalog", options=available_catalogs, key="vector_catalog")
            
            with col2:
                if selected_catalog:
                    available_schemas = get_schemas(client, selected_catalog)
                    selected_schema = st.selectbox("Select Schema", options=available_schemas, key="vector_schema")
                else:
                    selected_schema = None
                    st.selectbox("Select Schema", options=[], disabled=True, key="vector_schema_disabled")
            
            if selected_catalog and selected_schema:
                st.divider()
                
                # Vector Search Endpoint Setup
                st.subheader("🚀 Vector Search Endpoint")
                col1, col2 = st.columns([2, 1])
                with col1:
                    endpoint_name = st.text_input("Vector Search Endpoint Name", value="vector_search_endpoint")
                with col2:
                    if st.button("🚀 Create Endpoint"):
                        clear_current_step_messages()
                        create_vector_search_endpoint(client, endpoint_name)
                
                st.divider()
                
                # Step 1: Convert Files to Tables
                st.subheader("📋 Step 1: Convert Files to Tables")
                
                # Get available volumes and files
                available_volumes = get_volumes_from_catalog_schema(client, selected_catalog, selected_schema)
                
                if available_volumes:
                    selected_volume = st.selectbox(
                        "Select Volume", 
                        options=available_volumes,
                        key="vectorization_volume_select"
                    )
                    
                    if selected_volume:
                        available_files = get_files_from_volume(client, selected_catalog, selected_schema, selected_volume)
                        
                        if available_files:
                            st.write(f"**Files in volume `{selected_volume}`:**")
                            
                            # Multi-select for files
                            selected_files = []
                            
                            # Select all checkbox for files
                            select_all_files = st.checkbox("Select All Files", key="select_all_files")
                            
                            for i, file_info in enumerate(available_files):
                                checkbox_key = f"select_file_for_vectorization_{i}_{file_info['name']}"
                                is_selected = select_all_files or st.checkbox(
                                    f"📄 {file_info['name']} ({file_info['type']}) - {file_info['size']} bytes",
                                    key=checkbox_key,
                                    value=select_all_files
                                )
                                if is_selected:
                                    selected_files.append(file_info)
                            
                            if selected_files:
                                st.success(f"✅ Selected {len(selected_files)} file(s) for table creation")
                                
                                if st.button("📋 Create Tables from Selected Files", use_container_width=True):
                                    clear_current_step_messages()
                                    
                                    progress_bar = st.progress(0)
                                    status_container = st.empty()
                                    
                                    created_tables = create_text_table_from_volume_files(
                                        client, selected_files, selected_catalog, selected_schema,
                                        progress_bar, status_container
                                    )
                                    
                                    if created_tables:
                                        st.session_state['created_tables'] = created_tables
                        else:
                            st.info("📂 No files found in the selected volume")
                else:
                    st.info("📂 No volumes found in the selected catalog/schema")
                
                st.divider()
                
                # Step 2: Vectorize Tables
                st.subheader("🔍 Step 2: Vectorize Tables")
                
                # Get available tables (including newly created ones)
                available_tables = get_tables_from_catalog_schema(client, selected_catalog, selected_schema)
                
                # Filter for raw_text_parsed tables
                parsed_tables = [t for t in available_tables if t['name'].startswith('raw_text_parsed_')]
                
                if parsed_tables:
                    st.write("**Available parsed tables for vectorization:**")
                    
                    selected_table = st.selectbox(
                        "Select Table to Vectorize",
                        options=[t['name'] for t in parsed_tables],
                        key="table_for_vectorization"
                    )
                    
                    if selected_table:
                        # Get table columns
                        table_columns = get_table_columns(client, selected_catalog, selected_schema, selected_table)
                        
                        if table_columns:
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                primary_key = st.selectbox(
                                    "Primary Key Column",
                                    options=[col['name'] for col in table_columns],
                                    key="primary_key_select"
                                )
                                
                                sync_column = st.selectbox(
                                    "Column to Sync",
                                    options=[col['name'] for col in table_columns],
                                    key="sync_column_select"
                                )
                            
                            with col2:
                                embedding_column = st.selectbox(
                                    "Embedding Source Column",
                                    options=[col['name'] for col in table_columns if 'text' in col['type'].lower() or 'string' in col['type'].lower()],
                                    key="embedding_column_select"
                                )
                                
                                # Get available embedding models
                                embedding_models = get_embedding_models(client)
                                embedding_model = st.selectbox(
                                    "Embedding Model",
                                    options=embedding_models,
                                    key="embedding_model_select"
                                )
                            
                            with col3:
                                # Get available vector search endpoints
                                vector_endpoints = get_vector_search_endpoints(client)
                                if not vector_endpoints:
                                    vector_endpoints = [endpoint_name]  # Use the one we're creating
                                
                                vector_endpoint = st.selectbox(
                                    "Vector Search Endpoint",
                                    options=vector_endpoints,
                                    key="vector_endpoint_select"
                                )
                            
                            st.write("**Configuration Summary:**")
                            config_col1, config_col2 = st.columns(2)
                            with config_col1:
                                st.write(f"**Table:** `{selected_catalog}.{selected_schema}.{selected_table}`")
                                st.write(f"**Primary Key:** `{primary_key}`")
                                st.write(f"**Sync Column:** `{sync_column}`")
                            with config_col2:
                                st.write(f"**Embedding Column:** `{embedding_column}`")
                                st.write(f"**Embedding Model:** `{embedding_model}`")
                                st.write(f"**Vector Endpoint:** `{vector_endpoint}`")
                            
                            if st.button("🔍 Create Vector Index", use_container_width=True):
                                clear_current_step_messages()
                                
                                with st.spinner("Creating vector index..."):
                                    vector_index_name = create_vector_index(
                                        client, selected_table, selected_catalog, selected_schema,
                                        primary_key, sync_column, embedding_column,
                                        embedding_model, vector_endpoint
                                    )
                                
                                if vector_index_name:
                                    # Add to session state
                                    if 'vectorized_indexes' not in st.session_state:
                                        st.session_state.vectorized_indexes = []
                                    
                                    st.session_state.vectorized_indexes.append({
                                        'name': f"index_{selected_table}",
                                        'full_name': vector_index_name,
                                        'source_table': selected_table,
                                        'catalog': selected_catalog,
                                        'schema': selected_schema,
                                        'primary_key': primary_key,
                                        'embedding_column': embedding_column,
                                        'embedding_model': embedding_model,
                                        'endpoint': vector_endpoint
                                    })
                                    
                                    st.success(f"✅ Vector index created successfully: {vector_index_name}")
                        else:
                            st.error("❌ Could not retrieve table columns")
                else:
                    st.info("📋 No parsed tables found. Please create tables from files first.")
                
                # Display existing vector indexes for this catalog/schema
                if hasattr(st.session_state, 'vectorized_indexes'):
                    catalog_indexes = [idx for idx in st.session_state.vectorized_indexes 
                                     if idx.get('catalog') == selected_catalog and idx.get('schema') == selected_schema]
                    
                    if catalog_indexes:
                        st.divider()
                        st.subheader(f"🔍 Vector Indexes in {selected_catalog}.{selected_schema}")
                        
                        for idx in catalog_indexes:
                            with st.expander(f"📊 {idx['name']} (from {idx['source_table']})"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write(f"**Source Table:** {idx['source_table']}")
                                    st.write(f"**Primary Key:** {idx['primary_key']}")
                                    st.write(f"**Embedding Column:** {idx['embedding_column']}")
                                
                                with col2:
                                    st.code(f"Index: {idx['full_name']}", language="text")
                                    st.write(f"**Embedding Model:** {idx['embedding_model']}")
                                    st.write(f"**Endpoint:** {idx['endpoint']}")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")