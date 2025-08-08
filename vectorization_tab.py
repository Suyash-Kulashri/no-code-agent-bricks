import streamlit as st
from databricks.sdk import WorkspaceClient
from vectorization_utils import (
    get_catalogs, get_schemas, get_tables_from_catalog_schema, get_table_columns,
    get_vector_search_endpoints, get_embedding_models, create_vector_search_endpoint,
    create_vector_index, clear_current_step_messages, add_status_message,
    get_sql_warehouses, start_sql_warehouse, get_warehouse_status,
    get_serving_endpoints, render_enhanced_cdf_check
)
import time

def render_vectorization_tab(workspace_url, token):
    """Render the enhanced Data Vectorization tab"""
    st.subheader("📊 Table to Vector Index")
    st.write("Select a table with parsed text and create a Vector Search index from its content.")

    if not (workspace_url and token):
        st.warning("⚠️ Please configure Databricks connection in the main settings.")
        return
        
    try:
        client = WorkspaceClient(host=workspace_url, token=token)
        
        # --- Step 0: SQL Warehouse Management ---
        st.markdown("### 🏭 Step 0: SQL Warehouse Setup")
        
        warehouses = get_sql_warehouses(client)
        if not warehouses:
            st.error("❌ No SQL warehouses found in your workspace.")
            st.info("Please create a SQL warehouse in your Databricks workspace first.")
            return
        
        col1, col2 = st.columns([2, 1])
        with col1:
            warehouse_options = [f"{w['name']} ({w['id']}) - {w['state']}" for w in warehouses]
            selected_warehouse_idx = st.selectbox(
                "Select SQL Warehouse",
                range(len(warehouse_options)),
                format_func=lambda x: warehouse_options[x],
                key="sql_warehouse_select"
            )
            selected_warehouse = warehouses[selected_warehouse_idx]
        
        with col2:
            if selected_warehouse['state'] != 'RUNNING':
                if st.button(f"🚀 Start Warehouse", help="Start the selected SQL warehouse"):
                    with st.spinner("Starting SQL warehouse..."):
                        if start_sql_warehouse(client, selected_warehouse['id']):
                            st.success("✅ Warehouse start initiated")
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error("❌ Failed to start warehouse")
            else:
                st.success("✅ Warehouse is running")
        
        if selected_warehouse['state'] not in ['RUNNING', 'STARTING']:
            st.warning("⚠️ SQL warehouse must be running to proceed. Please start it above.")
            return
        
        st.divider()
        
        # --- Step 1: Select Catalog, Schema, and Table ---
        st.markdown("### 🎯 Step 1: Select Source Table")
        
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
        
        if not selected_catalog or not selected_schema:
            st.info("Please select a catalog and schema to see available tables.")
            return

        available_tables = get_tables_from_catalog_schema(client, selected_catalog, selected_schema)
        if not available_tables:
            st.warning("No tables found in the selected schema.")
            return
            
        selected_table_info = st.selectbox(
            "Select Table to Vectorize",
            options=available_tables,
            format_func=lambda t: t['name'],
            key="table_for_vectorization",
            help="Choose a table that contains parsed document chunks."
        )

        if not selected_table_info:
            return
        
        selected_table_name = selected_table_info['name']
        st.success(f"Selected table: `{selected_table_name}`")
        
        st.divider()

        # --- Step 2: Configure Vector Index Structure ---
        st.markdown("### ⚙️ Step 2: Index Structure Configuration")
        
        # Add a button to manually refresh table columns
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔄 Refresh Columns", help="Retry loading table columns"):
                clear_current_step_messages()
                st.rerun()
        
        table_columns = get_table_columns(client, selected_catalog, selected_schema, selected_table_name, selected_warehouse['id'])
        
        if not table_columns:
            st.error("❌ Could not retrieve columns for the selected table.")
            st.info("""
            **Troubleshooting steps:**
            1. Ensure the SQL warehouse is running
            2. Verify you have SELECT permissions on the table
            3. Check if the table exists and is accessible
            4. Try clicking 'Refresh Columns' above
            """)
            return
        
        st.success(f"✅ Found {len(table_columns)} columns in the table")
        
        # Index name input
        index_name = st.text_input(
            "Index Name",
            value=f"{selected_table_name}_vector_index",
            help="Enter a name for your new vector search index",
            placeholder="my_vector_index"
        )
        
        # Primary key selection
        primary_key = st.selectbox(
            "Primary Key",
            options=[col['name'] for col in table_columns],
            key="primary_key_select",
            help="Select a column with unique IDs for each text chunk."
        )
        
        # Columns to sync
        st.markdown("**Columns to sync**")
        sync_all_columns = st.checkbox("Sync all columns", value=True, help="Include all table columns in the index")
        
        selected_columns = []
        if not sync_all_columns:
            selected_columns = st.multiselect(
                "Select specific columns to sync",
                options=[col['name'] for col in table_columns],
                default=[col['name'] for col in table_columns],
                help="Choose which columns to include in the vector index"
            )
        
        st.divider()
        
        # --- Step 3: Embeddings Configuration ---
        st.markdown("### 🧠 Step 3: Embeddings Configuration")
        
        # Embedding source selection
        embedding_source = st.radio(
            "Embedding source",
            options=["Compute embeddings", "Use existing embedding column"],
            index=0,
            help="Choose whether to compute new embeddings or use pre-computed ones"
        )
        
        if embedding_source == "Compute embeddings":
            # Embedding source column - filter text columns safely
            text_columns = []
            for col in table_columns:
                col_type_str = str(col['type']).lower()
                if any(text_type in col_type_str for text_type in ['string', 'text', 'varchar', 'char']):
                    text_columns.append(col['name'])
            
            if not text_columns:
                st.warning("No suitable text columns found for computing embeddings")
                return
                
            embedding_column = st.selectbox(
                "Embedding source column",
                options=text_columns,
                key="embedding_column_select",
                help="Select the column containing the text to be vectorized."
            )
            
            # Embedding model selection
            serving_endpoints = get_serving_endpoints(client)
            embedding_models = get_embedding_models(client)
            all_models = serving_endpoints + embedding_models
            
            embedding_model = st.selectbox(
                "Embedding model",
                options=all_models,
                key="embedding_model_select",
                help="Select the model to generate embeddings"
            )
            
            # Sync computed embeddings
            sync_embeddings = st.checkbox(
                "Sync computed embeddings",
                value=True,
                help="Automatically sync embeddings when source data changes"
            )
        else:
            # For existing embedding column - filter vector/array columns safely
            vector_columns = []
            for col in table_columns:
                col_type_str = str(col['type']).lower()
                if any(vector_type in col_type_str for vector_type in ['array', 'vector', 'float', 'double']):
                    vector_columns.append(col['name'])
            
            if not vector_columns:
                st.warning("No suitable vector/array columns found for existing embeddings")
                return
            
            embedding_column = st.selectbox(
                "Select existing embedding column",
                options=vector_columns,
                help="Choose the column that already contains embeddings"
            )
            embedding_model = None
            sync_embeddings = False
        
        st.divider()
        
        # --- Step 4: Compute Resources ---
        st.markdown("### 💻 Step 4: Compute Resources")
        
        # Vector search endpoint
        vector_endpoints = get_vector_search_endpoints(client)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if vector_endpoints:
                vector_endpoint = st.selectbox(
                    "Vector search endpoint",
                    options=vector_endpoints,
                    help="Select an existing vector search endpoint"
                )
            else:
                vector_endpoint = st.text_input(
                    "Vector search endpoint name",
                    value="vector_search_endpoint",
                    help="Name for the new vector search endpoint to be created"
                )
        
        with col2:
            if not vector_endpoints:
                st.info("No existing endpoints found. A new one will be created.")
        
        # Sync mode
        sync_mode = st.radio(
            "Sync mode",
            options=["Triggered", "Continuous"],
            index=0,
            help="Triggered: Manual sync | Continuous: Automatic sync on data changes"
        )
        
        # Show column details
        if table_columns:
            with st.expander(f"📋 Table Columns ({len(table_columns)} found)", expanded=False):
                for col in table_columns:
                    st.write(f"• **{col['name']}** - `{col['type']}`")
        
        st.divider()
        
        # --- Step 4.5: Change Data Feed Setup (Optional) ---
        st.markdown("### 🔧 Step 4.5: Change Data Feed Setup (Optional)")
        st.write("Change Data Feed (CDF) allows automatic syncing of the index when the table changes. "
                "If you skip CDF, the index will only update when you manually trigger a sync.")

        from vectorization_utils import check_change_data_feed_status_fixed

        with st.spinner("Checking Change Data Feed status..."):
            cdf_ready = check_change_data_feed_status_fixed(
                client, selected_catalog, selected_schema, selected_table_name, selected_warehouse['id']
            )

        if not cdf_ready:
            st.warning("⚠️ CDF is not enabled for this table. Proceeding without it — index will not auto-sync.")
            sync_mode = "Triggered"  # Force triggered mode without CDF
        else:
            st.success("✅ CDF is enabled — you can use Continuous sync mode.")
        
        # --- Step 5: Create Index ---
        st.markdown("### 🚀 Step 5: Create Vector Index")
        
        # Validation checks
        validation_errors = []
        if not primary_key:
            validation_errors.append("Please select a Primary Key")
        if not embedding_column:
            validation_errors.append("Please select an Embedding Column")
        if not index_name.strip():
            validation_errors.append("Please provide an Index Name")
        if not vector_endpoint:
            validation_errors.append("Please specify a Vector Search Endpoint")
        
        if validation_errors:
            for error in validation_errors:
                st.error(f"❌ {error}")
        else:
            # Show configuration summary
            with st.expander("📋 Configuration Summary", expanded=True):
                st.write(f"**Table:** `{selected_catalog}.{selected_schema}.{selected_table_name}`")
                st.write(f"**Index Name:** `{index_name}`")
                st.write(f"**Primary Key:** `{primary_key}`")
                st.write(f"**Embedding Column:** `{embedding_column}`")
                if embedding_model:
                    st.write(f"**Embedding Model:** `{embedding_model}`")
                st.write(f"**Vector Endpoint:** `{vector_endpoint}`")
                st.write(f"**Sync Mode:** `{sync_mode}`")
                if not sync_all_columns:
                    st.write(f"**Columns to Sync:** {', '.join(selected_columns)}")
                else:
                    st.write(f"**Columns to Sync:** All columns")
            
            if st.button("🔍 Create Vector Index", use_container_width=True, type="primary"):
                clear_current_step_messages()
                with st.spinner("Creating Vector Index... This may take several minutes."):
                    
                    # First, ensure the endpoint exists or create it
                    add_status_message(f"🔄 Ensuring endpoint '{vector_endpoint}' exists...")
                    endpoint_created = create_vector_search_endpoint(client, vector_endpoint)
                    
                    if endpoint_created:
                        # Wait a moment and verify endpoint is accessible
                        add_status_message("🔄 Verifying endpoint accessibility...")
                        time.sleep(2)  # Give endpoint time to initialize
                        
                        # Verify endpoint exists in the list
                        current_endpoints = get_vector_search_endpoints(client)
                        if vector_endpoint not in current_endpoints:
                            add_status_message(f"⚠️ Endpoint '{vector_endpoint}' created but not yet visible. Proceeding anyway...", False)
                        else:
                            add_status_message(f"✅ Endpoint '{vector_endpoint}' verified and ready")
                        
                        # Prepare columns to sync
                        columns_to_sync = selected_columns if not sync_all_columns else None
                        
                        # Create the index
                        add_status_message(f"✅ Endpoint ready. Creating index '{index_name}'...")
                        vector_index_name = create_vector_index(
                            client, 
                            catalog_name=selected_catalog,
                            schema_name=selected_schema,
                            table_name=selected_table_name,
                            index_name=index_name,
                            primary_key=primary_key,
                            embedding_column=embedding_column,
                            embedding_model=embedding_model,
                            vector_endpoint=vector_endpoint,
                            sync_mode=sync_mode.upper(),
                            columns_to_sync=columns_to_sync,
                            compute_embeddings=(embedding_source == "Compute embeddings")
                        )
                        
                        if vector_index_name:
                            st.success(f"✅ Successfully created Vector Index: `{vector_index_name}`")
                            st.info("📊 The index may take several minutes to become fully available and synced.")
                            st.info(f"🔗 You can monitor the index status in the Databricks Vector Search console.")
                            
                            # Show status messages in an expandable section
                            if 'current_step_messages' in st.session_state and st.session_state.current_step_messages:
                                with st.expander("📋 Detailed Creation Log", expanded=False):
                                    for msg in st.session_state.current_step_messages:
                                        if msg['is_success']:
                                            st.success(msg['message'])
                                        else:
                                            st.warning(msg['message'])
                            
                            st.balloons()
                        else:
                            st.error("❌ Failed to create vector index. Check the error messages below.")
                            # Show error messages
                            if 'current_step_messages' in st.session_state and st.session_state.current_step_messages:
                                with st.expander("🔍 Error Details", expanded=True):
                                    for msg in st.session_state.current_step_messages:
                                        if msg['is_success']:
                                            st.success(msg['message'])
                                        else:
                                            st.error(msg['message'])
                    else:
                        st.error("❌ Could not create or verify the Vector Search Endpoint. Cannot proceed.")
                        st.error("**Possible causes:**")
                        st.error("• Insufficient permissions for Vector Search")
                        st.error("• Vector Search not enabled in your workspace")
                        st.error("• Network connectivity issues")
                        
                        # Show detailed error messages
                        if 'current_step_messages' in st.session_state and st.session_state.current_step_messages:
                            with st.expander("🔍 Error Details", expanded=True):
                                for msg in st.session_state.current_step_messages:
                                    if msg['is_success']:
                                        st.success(msg['message'])
                                    else:
                                        st.error(msg['message'])

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.info("Please check your Databricks connection and permissions.")