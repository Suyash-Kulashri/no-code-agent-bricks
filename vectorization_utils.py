import streamlit as st
from databricks.sdk import WorkspaceClient
import requests
import time

# This utility file contains functions needed for the enhanced vectorization tab.

def add_status_message(message: str, is_success: bool = True):
    """Add a status message to current step"""
    if 'current_step_messages' not in st.session_state:
        st.session_state.current_step_messages = []
    st.session_state.current_step_messages.append({
        'message': message,
        'is_success': is_success,
        'timestamp': time.time()
    })

def clear_current_step_messages():
    """Clear messages from current step"""
    st.session_state.current_step_messages = []

def get_catalogs(client: WorkspaceClient):
    """Get list of all catalogs"""
    try:
        return [catalog.name for catalog in client.catalogs.list()]
    except Exception as e:
        add_status_message(f"❌ Error fetching catalogs: {str(e)}", False)
        return []

def get_schemas(client: WorkspaceClient, catalog_name: str):
    """Get list of schemas in a catalog"""
    try:
        return [schema.name for schema in client.schemas.list(catalog_name=catalog_name)]
    except Exception as e:
        add_status_message(f"❌ Error fetching schemas: {str(e)}", False)
        return []

def get_sql_warehouses(client: WorkspaceClient):
    """Get list of all SQL warehouses with their status"""
    try:
        warehouses = []
        for warehouse in client.warehouses.list():
            warehouses.append({
                'id': warehouse.id,
                'name': warehouse.name,
                'state': warehouse.state.value if warehouse.state else 'UNKNOWN',
                'cluster_size': warehouse.cluster_size,
                'auto_stop_mins': warehouse.auto_stop_mins
            })
        return warehouses
    except Exception as e:
        add_status_message(f"❌ Error fetching SQL warehouses: {str(e)}", False)
        return []

def start_sql_warehouse(client: WorkspaceClient, warehouse_id: str):
    """Start a SQL warehouse"""
    try:
        client.warehouses.start(id=warehouse_id)
        add_status_message(f"✅ SQL warehouse {warehouse_id} start initiated")
        return True
    except Exception as e:
        add_status_message(f"❌ Error starting SQL warehouse: {str(e)}", False)
        return False

def get_warehouse_status(client: WorkspaceClient, warehouse_id: str):
    """Get the current status of a SQL warehouse"""
    try:
        warehouse = client.warehouses.get(id=warehouse_id)
        return warehouse.state.value if warehouse.state else 'UNKNOWN'
    except Exception as e:
        add_status_message(f"❌ Error getting warehouse status: {str(e)}", False)
        return 'ERROR'

def get_warehouse_id(client: WorkspaceClient):
    """Get the first available and running SQL warehouse ID"""
    try:
        warehouses = client.warehouses.list()
        # Prefer a running warehouse
        for warehouse in warehouses:
            if warehouse.state == 'RUNNING':
                return warehouse.id
        # Fallback to the first available if none are running
        warehouses_list = list(warehouses)
        if warehouses_list:
            return warehouses_list[0].id # It will auto-start if needed
        # If no warehouses exist at all
        raise Exception("No SQL warehouses found in the workspace.")
    except Exception as e:
        # Re-raise with a more user-friendly message
        raise Exception(f"Could not retrieve a SQL Warehouse ID. Please ensure one exists. Error: {str(e)}")

def get_tables_from_catalog_schema(client: WorkspaceClient, catalog_name: str, schema_name: str):
    """Get all tables from a specific catalog and schema"""
    try:
        tables = list(client.tables.list(catalog_name=catalog_name, schema_name=schema_name))
        return [{'name': table.name, 'full_name': table.full_name} for table in tables]
    except Exception as e:
        add_status_message(f"❌ Error fetching tables: {str(e)}", False)
        return []

def get_table_columns(client: WorkspaceClient, catalog_name: str, schema_name: str, table_name: str, warehouse_id: str = None):
    """Get column information for a table with multiple fallback methods."""
    if not warehouse_id:
        try:
            warehouse_id = get_warehouse_id(client)
            add_status_message(f"🔄 Using warehouse: {warehouse_id}")
        except Exception as e:
            st.error(f"🔴 **Action Required:** Could not find a SQL Warehouse. {e}")
            st.info("Please ensure a SQL Warehouse exists and you have permissions to use it.")
            return []

    full_table_name = f"`{catalog_name}`.`{schema_name}`.`{table_name}`"
    
    # Method 1: Try DESCRIBE TABLE (most common)
    columns = _try_describe_table(client, warehouse_id, full_table_name)
    if columns:
        return columns
    
    # Method 2: Try DESCRIBE TABLE EXTENDED
    columns = _try_describe_table_extended(client, warehouse_id, full_table_name)
    if columns:
        return columns
    
    # Method 3: Try using INFORMATION_SCHEMA
    columns = _try_information_schema(client, warehouse_id, catalog_name, schema_name, table_name)
    if columns:
        return columns
    
    # Method 4: Try using Databricks SDK directly
    columns = _try_sdk_table_info(client, catalog_name, schema_name, table_name)
    if columns:
        return columns
    
    add_status_message("❌ All methods to retrieve table columns have failed.", False)
    return []

def _try_describe_table(client: WorkspaceClient, warehouse_id: str, full_table_name: str):
    """Try standard DESCRIBE TABLE command"""
    try:
        add_status_message(f"🔄 Trying DESCRIBE TABLE for {full_table_name}")
        describe_sql = f"DESCRIBE TABLE {full_table_name}"
        
        result = client.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            statement=describe_sql,
            wait_timeout="60s"  # Increased timeout
        )
        
        return _parse_describe_result(result)
    except Exception as e:
        add_status_message(f"⚠️ DESCRIBE TABLE failed: {str(e)}", False)
        return []

def _try_describe_table_extended(client: WorkspaceClient, warehouse_id: str, full_table_name: str):
    """Try DESCRIBE TABLE EXTENDED command"""
    try:
        add_status_message(f"🔄 Trying DESCRIBE TABLE EXTENDED for {full_table_name}")
        describe_sql = f"DESCRIBE TABLE EXTENDED {full_table_name}"
        
        result = client.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            statement=describe_sql,
            wait_timeout="60s"
        )
        
        return _parse_describe_result(result)
    except Exception as e:
        add_status_message(f"⚠️ DESCRIBE TABLE EXTENDED failed: {str(e)}", False)
        return []

def _try_information_schema(client: WorkspaceClient, warehouse_id: str, catalog_name: str, schema_name: str, table_name: str):
    """Try querying INFORMATION_SCHEMA.COLUMNS"""
    try:
        add_status_message(f"🔄 Trying INFORMATION_SCHEMA for {catalog_name}.{schema_name}.{table_name}")
        info_sql = f"""
        SELECT column_name, data_type 
        FROM {catalog_name}.information_schema.columns 
        WHERE table_catalog = '{catalog_name}' 
          AND table_schema = '{schema_name}' 
          AND table_name = '{table_name}'
        ORDER BY ordinal_position
        """
        
        result = client.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            statement=info_sql,
            wait_timeout="60s"
        )
        
        columns = []
        if result.status.state == 'SUCCEEDED' and result.result and result.result.data_array:
            for row in result.result.data_array:
                if row and len(row) >= 2:
                    columns.append({'name': row[0], 'type': row[1]})
            add_status_message(f"✅ Retrieved {len(columns)} columns via INFORMATION_SCHEMA")
            return columns
        return []
        
    except Exception as e:
        add_status_message(f"⚠️ INFORMATION_SCHEMA query failed: {str(e)}", False)
        return []

def _try_sdk_table_info(client: WorkspaceClient, catalog_name: str, schema_name: str, table_name: str):
    """Try using Databricks SDK table info directly"""
    try:
        add_status_message(f"🔄 Trying SDK table.get for {catalog_name}.{schema_name}.{table_name}")
        full_name = f"{catalog_name}.{schema_name}.{table_name}"
        
        table_info = client.tables.get(full_name=full_name)
        
        if table_info and table_info.columns:
            columns = []
            for col in table_info.columns:
                # Handle different type representations
                col_type = ""
                if hasattr(col, 'type_name') and col.type_name:
                    col_type = str(col.type_name)
                elif hasattr(col, 'type_text') and col.type_text:
                    col_type = str(col.type_text)
                elif hasattr(col, 'type_json') and col.type_json:
                    col_type = str(col.type_json)
                else:
                    col_type = "unknown"
                
                columns.append({
                    'name': col.name,
                    'type': col_type
                })
            add_status_message(f"✅ Retrieved {len(columns)} columns via SDK")
            return columns
        return []
        
    except Exception as e:
        add_status_message(f"⚠️ SDK table.get failed: {str(e)}", False)
        return []

def _parse_describe_result(result):
    """Parse the result from DESCRIBE TABLE commands"""
    columns = []
    if result.status.state == 'SUCCEEDED' and result.result and result.result.data_array:
        for row in result.result.data_array:
            # Filter out comments, partition info, and empty rows
            if (row and len(row) >= 2 and 
                not row[0].startswith('#') and 
                row[0].strip() != '' and
                not row[0].lower().startswith('partition') and
                row[0] != 'col_name'):  # Skip header row if present
                columns.append({'name': row[0].strip(), 'type': row[1].strip()})
        
        if columns:
            add_status_message(f"✅ Retrieved {len(columns)} columns")
            return columns
    elif result.status.state == 'FAILED':
        error_msg = result.status.error.message if result.status.error else "Unknown error"
        add_status_message(f"❌ SQL query failed: {error_msg}", False)
    
    return []

def get_vector_search_endpoints(client: WorkspaceClient):
    """Get list of available vector search endpoints with multiple methods"""
    # Method 1: Try REST API
    endpoints = _get_endpoints_via_api(client)
    if endpoints:
        return endpoints
    
    # Method 2: Try SDK (fallback)
    return _get_endpoints_via_sdk(client)

def _get_endpoints_via_api(client: WorkspaceClient):
    """Get endpoints via REST API"""
    try:
        response = requests.get(
            f"{client.config.host}/api/2.0/vector-search/endpoints",
            headers={"Authorization": f"Bearer {client.config.token}"},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            endpoints = [ep.get("name", "") for ep in data.get("endpoints", []) if ep and ep.get("name")]
            add_status_message(f"✅ Found {len(endpoints)} vector search endpoints via API")
            return endpoints
        else:
            add_status_message(f"⚠️ API call failed with status {response.status_code}", False)
            return []
    except Exception as e:
        add_status_message(f"⚠️ REST API endpoint listing failed: {str(e)}", False)
        return []

def _get_endpoints_via_sdk(client: WorkspaceClient):
    """Get endpoints via SDK (fallback)"""
    try:
        endpoints = []
        for endpoint in client.vector_search_endpoints.list_endpoints():
            if endpoint and endpoint.name:
                endpoints.append(endpoint.name)
        add_status_message(f"✅ Found {len(endpoints)} vector search endpoints via SDK")
        return endpoints
    except Exception as e:
        add_status_message(f"⚠️ SDK endpoint listing failed: {str(e)}", False)
        return []

def get_serving_endpoints(client: WorkspaceClient):
    """Get list of available serving endpoints for embeddings"""
    try:
        serving_endpoints = []
        for endpoint in client.serving_endpoints.list():
            if endpoint.name and 'embed' in endpoint.name.lower():
                serving_endpoints.append(endpoint.name)
        return serving_endpoints
    except Exception as e:
        add_status_message(f"⚠️ Could not fetch serving endpoints: {str(e)}", False)
        return []

def get_embedding_models(client: WorkspaceClient):
    """Get list of available embedding models"""
    return [
        "databricks-bge-large-en",
        "databricks-gte-large-en", 
        "text-embedding-ada-002"
    ]

def create_vector_search_endpoint(client: WorkspaceClient, endpoint_name: str):
    """Create a vector search endpoint if it doesn't exist"""
    try:
        # First, check if the endpoint already exists
        existing_endpoints = get_vector_search_endpoints(client)
        if endpoint_name in existing_endpoints:
            add_status_message(f"ℹ️ Vector search endpoint '{endpoint_name}' already exists.")
            return True
        
        # Try to create the endpoint using the SDK
        add_status_message(f"🔄 Creating vector search endpoint '{endpoint_name}'...")
        try:
            client.vector_search_endpoints.create_endpoint(
                name=endpoint_name, 
                endpoint_type="STANDARD"
            )
            add_status_message(f"✅ Started creation of endpoint: {endpoint_name}. This may take several minutes.")
            return True
        except Exception as sdk_error:
            add_status_message(f"⚠️ SDK endpoint creation failed: {str(sdk_error)}", False)
            
            # Fallback: Try using REST API directly
            return _create_endpoint_via_api(client, endpoint_name)
            
    except Exception as e:
        if "already exists" in str(e).lower():
            add_status_message(f"ℹ️ Vector search endpoint '{endpoint_name}' already exists.")
            return True
        add_status_message(f"❌ Error with vector search endpoint: {str(e)}", False)
        return False

def _create_endpoint_via_api(client: WorkspaceClient, endpoint_name: str):
    """Fallback method to create endpoint via REST API"""
    try:
        add_status_message(f"🔄 Trying REST API to create endpoint '{endpoint_name}'...")
        
        # Prepare the request payload
        payload = {
            "name": endpoint_name,
            "endpoint_type": "STANDARD"
        }
        
        # Make the REST API call
        response = requests.post(
            f"{client.config.host}/api/2.0/vector-search/endpoints",
            headers={
                "Authorization": f"Bearer {client.config.token}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            add_status_message(f"✅ Successfully created endpoint '{endpoint_name}' via REST API")
            return True
        elif response.status_code == 409:
            add_status_message(f"ℹ️ Endpoint '{endpoint_name}' already exists (REST API)")
            return True
        else:
            error_detail = ""
            try:
                error_data = response.json()
                error_detail = error_data.get('message', f"Status code: {response.status_code}")
            except:
                error_detail = f"Status code: {response.status_code}"
            
            add_status_message(f"❌ REST API endpoint creation failed: {error_detail}", False)
            return False
            
    except Exception as api_error:
        add_status_message(f"❌ REST API call failed: {str(api_error)}", False)
        return False

def create_vector_index(client: WorkspaceClient, catalog_name: str, schema_name: str, table_name: str,
                      index_name: str, primary_key: str, embedding_column: str,
                      embedding_model: str, vector_endpoint: str, sync_mode: str = "TRIGGERED",
                      columns_to_sync: list = None, compute_embeddings: bool = True):
    """Create vector search index for a table with enhanced options"""
    try:
        full_index_name = f"{catalog_name}.{schema_name}.{index_name}"
        full_table_name = f"{catalog_name}.{schema_name}.{table_name}"

        vs_client = client.vector_search_indexes
        
        # Import the required classes for proper object creation
        from databricks.sdk.service.vectorsearch import (
            DeltaSyncVectorIndexSpecRequest,
            EmbeddingSourceColumn,
            EmbeddingVectorColumn
        )
        
        add_status_message(f"🔄 Preparing index specification...")
        
        # Build the index specification based on whether we compute embeddings or use existing ones
        if compute_embeddings:
            # For computed embeddings
            embedding_source_columns = [
                EmbeddingSourceColumn(
                    name=embedding_column,
                    embedding_model_endpoint_name=embedding_model
                )
            ]
            
            delta_sync_spec = DeltaSyncVectorIndexSpecRequest(
                source_table=full_table_name,
                pipeline_type=sync_mode,
                embedding_source_columns=embedding_source_columns
            )
        else:
            # For existing embedding columns
            embedding_vector_columns = [
                EmbeddingVectorColumn(
                    name=embedding_column,
                    dimension=1536  # Default dimension, may need adjustment
                )
            ]
            
            delta_sync_spec = DeltaSyncVectorIndexSpecRequest(
                source_table=full_table_name,
                pipeline_type=sync_mode,
                embedding_vector_columns=embedding_vector_columns
            )
        
        # Add columns to sync if specified
        if columns_to_sync:
            delta_sync_spec.columns_to_sync = columns_to_sync
        
        add_status_message(f"🔄 Creating vector index '{full_index_name}'...")
        
        vs_client.create_index(
            name=full_index_name,
            endpoint_name=vector_endpoint,
            primary_key=primary_key,
            index_type="DELTA_SYNC",
            delta_sync_index_spec=delta_sync_spec
        )
        
        add_status_message(f"✅ Successfully initiated creation of vector index: {full_index_name}")
        return full_index_name
        
    except ImportError as ie:
        add_status_message(f"❌ Failed to import required Databricks SDK classes: {str(ie)}", False)
        # Fallback to REST API method
        return _create_index_via_api(client, catalog_name, schema_name, table_name, 
                                   index_name, primary_key, embedding_column, 
                                   embedding_model, vector_endpoint, sync_mode, 
                                   columns_to_sync, compute_embeddings)
        
    except Exception as e:
        if "already exists" in str(e).lower():
            add_status_message(f"ℹ️ Vector index '{index_name}' already exists.")
            return f"{catalog_name}.{schema_name}.{index_name}"
        add_status_message(f"❌ Error creating vector index via SDK: {str(e)}", False)
        
        # Fallback to REST API method
        return _create_index_via_api(client, catalog_name, schema_name, table_name, 
                                   index_name, primary_key, embedding_column, 
                                   embedding_model, vector_endpoint, sync_mode, 
                                   columns_to_sync, compute_embeddings)

def _create_index_via_api(client: WorkspaceClient, catalog_name: str, schema_name: str, table_name: str,
                         index_name: str, primary_key: str, embedding_column: str,
                         embedding_model: str, vector_endpoint: str, sync_mode: str,
                         columns_to_sync: list, compute_embeddings: bool):
    """Fallback method to create vector index via REST API"""
    try:
        add_status_message(f"🔄 Trying REST API to create vector index...")
        
        full_index_name = f"{catalog_name}.{schema_name}.{index_name}"
        full_table_name = f"{catalog_name}.{schema_name}.{table_name}"
        
        # Build the payload
        if compute_embeddings:
            delta_sync_spec = {
                "source_table": full_table_name,
                "pipeline_type": sync_mode,
                "embedding_source_columns": [
                    {
                        "name": embedding_column,
                        "embedding_model_endpoint_name": embedding_model
                    }
                ]
            }
        else:
            delta_sync_spec = {
                "source_table": full_table_name,
                "pipeline_type": sync_mode,
                "embedding_vector_columns": [
                    {
                        "name": embedding_column,
                        "dimension": 1536
                    }
                ]
            }
        
        if columns_to_sync:
            delta_sync_spec["columns_to_sync"] = columns_to_sync
        
        payload = {
            "name": full_index_name,
            "endpoint_name": vector_endpoint,
            "primary_key": primary_key,
            "index_type": "DELTA_SYNC",
            "delta_sync_index_spec": delta_sync_spec
        }
        
        response = requests.post(
            f"{client.config.host}/api/2.0/vector-search/indexes",
            headers={
                "Authorization": f"Bearer {client.config.token}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=60
        )
        
        if response.status_code in [200, 201]:
            add_status_message(f"✅ Successfully created vector index '{full_index_name}' via REST API")
            return full_index_name
        elif response.status_code == 409:
            add_status_message(f"ℹ️ Vector index '{index_name}' already exists (REST API)")
            return full_index_name
        else:
            error_detail = ""
            try:
                error_data = response.json()
                error_detail = error_data.get('message', f"Status code: {response.status_code}")
            except:
                error_detail = f"Status code: {response.status_code}, Response: {response.text}"
            
            add_status_message(f"❌ REST API index creation failed: {error_detail}", False)
            return None
            
    except Exception as api_error:
        add_status_message(f"❌ REST API index creation failed: {str(api_error)}", False)
        return None

# ===========================
# ENHANCED CDF FUNCTIONS
# ===========================

def enable_change_data_feed_fixed(client: WorkspaceClient, catalog_name: str, schema_name: str, table_name: str, warehouse_id: str):
    """Enable Change Data Feed on a Delta table - improved version with better error handling"""
    try:
        # Ensure warehouse is running first
        warehouse_status = get_warehouse_status(client, warehouse_id)
        if warehouse_status not in ['RUNNING']:
            add_status_message(f"⚠️ SQL warehouse is {warehouse_status}. Starting it...", False)
            if not start_sql_warehouse(client, warehouse_id):
                return False
            
            # Wait for warehouse to start
            for _ in range(12):  # Wait up to 60 seconds
                time.sleep(5)
                status = get_warehouse_status(client, warehouse_id)
                if status == 'RUNNING':
                    break
                add_status_message(f"🔄 Warehouse status: {status}")
            else:
                add_status_message("❌ Warehouse failed to start in time", False)
                return False
        
        # Use proper table name formatting
        full_table_name = f"{catalog_name}.{schema_name}.{table_name}"
        
        add_status_message(f"🔄 Enabling Change Data Feed for table {full_table_name}...")
        
        # Try different SQL variations
        sql_commands = [
            f"ALTER TABLE {full_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)",
            f"ALTER TABLE `{catalog_name}`.`{schema_name}`.`{table_name}` SET TBLPROPERTIES (delta.enableChangeDataFeed = true)",
            f"ALTER TABLE {catalog_name}.{schema_name}.{table_name} SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')"
        ]
        
        for i, sql_command in enumerate(sql_commands, 1):
            try:
                add_status_message(f"🔄 Attempt {i}: Executing CDF enable command...")
                
                result = client.statement_execution.execute_statement(
                    warehouse_id=warehouse_id,
                    statement=sql_command,
                    wait_timeout="45s"  # Increased timeout
                )
                
                if result.status.state == 'SUCCEEDED':
                    add_status_message(f"✅ Successfully enabled Change Data Feed for {full_table_name}")
                    
                    # Verify CDF is actually enabled
                    time.sleep(2)  # Give it a moment to propagate
                    if check_change_data_feed_status_fixed(client, catalog_name, schema_name, table_name, warehouse_id):
                        add_status_message("✅ CDF status verified as enabled")
                        return True
                    else:
                        add_status_message("⚠️ CDF command succeeded but status check failed", False)
                        return True  # Proceed anyway, might be a delay
                        
                elif result.status.state == 'FAILED':
                    error_msg = result.status.error.message if result.status.error else "Unknown error"
                    add_status_message(f"⚠️ Attempt {i} failed: {error_msg}", False)
                    if i == len(sql_commands):  # Last attempt
                        return False
                    continue
                    
            except Exception as cmd_error:
                add_status_message(f"⚠️ Attempt {i} exception: {str(cmd_error)}", False)
                if i == len(sql_commands):  # Last attempt
                    return False
                continue
        
        return False
        
    except Exception as e:
        add_status_message(f"❌ Error enabling Change Data Feed: {str(e)}", False)
        return False

def check_change_data_feed_status_fixed(client: WorkspaceClient, catalog_name: str, schema_name: str, table_name: str, warehouse_id: str):
    """Check if Change Data Feed is enabled on a table - improved version"""
    try:
        # Multiple methods to check CDF status
        methods = [
            ("SHOW TBLPROPERTIES", f"SHOW TBLPROPERTIES {catalog_name}.{schema_name}.{table_name}"),
            ("DESCRIBE TABLE EXTENDED", f"DESCRIBE TABLE EXTENDED {catalog_name}.{schema_name}.{table_name}"),
            ("SHOW TBLPROPERTIES with backticks", f"SHOW TBLPROPERTIES `{catalog_name}`.`{schema_name}`.`{table_name}`")
        ]
        
        for method_name, sql_command in methods:
            try:
                add_status_message(f"🔄 Checking CDF status using {method_name}...")
                
                result = client.statement_execution.execute_statement(
                    warehouse_id=warehouse_id,
                    statement=sql_command,
                    wait_timeout="30s"
                )
                
                if result.status.state == 'SUCCEEDED' and result.result and result.result.data_array:
                    for row in result.result.data_array:
                        if row and len(row) >= 2:
                            # Check for CDF property
                            if (row[0] and 'delta.enableChangeDataFeed' in str(row[0]) and 
                                row[1] and str(row[1]).lower() == 'true'):
                                add_status_message(f"✅ CDF is enabled (verified via {method_name})")
                                return True
                
                add_status_message(f"⚠️ {method_name} didn't show CDF as enabled")
                        
            except Exception as method_error:
                add_status_message(f"⚠️ {method_name} failed: {str(method_error)}", False)
                continue
        
        add_status_message("❌ Could not verify CDF status via any method", False)
        return False
        
    except Exception as e:
        add_status_message(f"❌ Error checking CDF status: {str(e)}", False)
        return False

def render_enhanced_cdf_check(client, selected_catalog, selected_schema, selected_table_name, selected_warehouse):
    """Enhanced CDF check and fix with better error handling"""
    
    st.markdown("**Change Data Feed (CDF) Status**")
    
    # First, check current status
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("🔍 Check CDF Status", help="Check if CDF is already enabled"):
            with st.spinner("Checking CDF status..."):
                is_enabled = check_change_data_feed_status_fixed(
                    client, selected_catalog, selected_schema, 
                    selected_table_name, selected_warehouse['id']
                )
                
                if is_enabled:
                    st.success("✅ Change Data Feed is already enabled!")
                    st.session_state.cdf_enabled = True
                else:
                    st.warning("⚠️ Change Data Feed is not enabled")
                    st.session_state.cdf_enabled = False
    
    # Show current status if we have it
    if hasattr(st.session_state, 'cdf_enabled'):
        if st.session_state.cdf_enabled:
            st.success("✅ CDF Status: Enabled")
            return True
        else:
            st.warning("⚠️ CDF Status: Not Enabled")
    
    # Show manual SQL option
    full_table_name = f"{selected_catalog}.{selected_schema}.{selected_table_name}"
    sql_command = f"ALTER TABLE {full_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
    
    with st.expander("📝 Manual SQL Command", expanded=False):
        st.info("Run this SQL command in your Databricks workspace:")
        st.code(sql_command, language="sql")
    
    # Provide automatic and manual options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔧 Enable CDF Automatically", type="primary", help="Attempt to enable CDF automatically"):
            clear_current_step_messages()
            with st.spinner("Enabling Change Data Feed..."):
                success = enable_change_data_feed_fixed(
                    client, selected_catalog, selected_schema, 
                    selected_table_name, selected_warehouse['id']
                )
                
                if success:
                    st.success("✅ Change Data Feed enabled successfully!")
                    st.session_state.cdf_enabled = True
                    
                    # Show status messages
                    if hasattr(st.session_state, 'current_step_messages') and st.session_state.current_step_messages:
                        with st.expander("📋 Enable Log", expanded=False):
                            for msg in st.session_state.current_step_messages:
                                if msg['is_success']:
                                    st.success(msg['message'])
                                else:
                                    st.warning(msg['message'])
                    
                    return True
                else:
                    st.error("❌ Automatic enable failed. Please check the error details below.")
                    
                    # Show error details
                    if hasattr(st.session_state, 'current_step_messages') and st.session_state.current_step_messages:
                        with st.expander("🔍 Error Details", expanded=True):
                            for msg in st.session_state.current_step_messages:
                                if msg['is_success']:
                                    st.success(msg['message'])
                                else:
                                    st.error(msg['message'])
                    
                    st.info("**Try these steps:**")
                    st.info("1. Run the manual SQL command above in Databricks")
                    st.info("2. Ensure your SQL warehouse is running")
                    st.info("3. Check you have ALTER permissions on the table")
                    
                    return False
    
    with col2:
        if st.button("✅ I've enabled CDF manually", type="secondary", help="Continue assuming CDF is enabled"):
            st.success("✅ Proceeding with vector index creation...")
            st.session_state.cdf_enabled = True
            return True
    
    with col3:
        if st.button("⏭️ Skip CDF Check", help="Proceed without CDF (may cause issues)"):
            st.warning("⚠️ Proceeding without CDF verification. Vector index sync may not work properly.")
            return True
    
    return False

def troubleshoot_cdf_issues(client, catalog_name, schema_name, table_name, warehouse_id):
    """Provide troubleshooting information for CDF issues"""
    st.markdown("### 🔧 CDF Troubleshooting")
    
    issues_and_solutions = [
        {
            "issue": "Table doesn't exist or not accessible",
            "check": f"SELECT COUNT(*) FROM {catalog_name}.{schema_name}.{table_name} LIMIT 1",
            "solution": "Verify table name and permissions"
        },
        {
            "issue": "Not a Delta table",
            "check": f"DESCRIBE DETAIL {catalog_name}.{schema_name}.{table_name}",
            "solution": "CDF only works with Delta tables"
        },
        {
            "issue": "Missing ALTER permissions",
            "check": f"SHOW GRANT ON TABLE {catalog_name}.{schema_name}.{table_name}",
            "solution": "Request ALTER permissions from admin"
        },
        {
            "issue": "SQL Warehouse not running",
            "check": "Check warehouse status",
            "solution": "Start the SQL warehouse first"
        }
    ]
    
    for item in issues_and_solutions:
        with st.expander(f"❓ {item['issue']}", expanded=False):
            st.code(item['check'], language="sql")
            st.info(f"**Solution:** {item['solution']}")

# ===========================
# LEGACY FUNCTIONS (DEPRECATED)
# ===========================

def enable_change_data_feed(client: WorkspaceClient, catalog_name: str, schema_name: str, table_name: str, warehouse_id: str = None):
    """Enable Change Data Feed on a Delta table - DEPRECATED: Use enable_change_data_feed_fixed instead"""
    return enable_change_data_feed_fixed(client, catalog_name, schema_name, table_name, warehouse_id)

def check_change_data_feed_status(client: WorkspaceClient, catalog_name: str, schema_name: str, table_name: str, warehouse_id: str = None):
    """Check if Change Data Feed is enabled on a table - DEPRECATED: Use check_change_data_feed_status_fixed instead"""
    if not warehouse_id:
        warehouse_id = get_warehouse_id(client)
    return check_change_data_feed_status_fixed(client, catalog_name, schema_name, table_name, warehouse_id)

def render_simple_cdf_check(client, selected_catalog, selected_schema, selected_table_name, selected_warehouse):
    """Simple CDF check and fix - DEPRECATED: Use render_enhanced_cdf_check instead"""
    return render_enhanced_cdf_check(client, selected_catalog, selected_schema, selected_table_name, selected_warehouse)