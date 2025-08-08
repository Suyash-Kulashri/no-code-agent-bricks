import streamlit as st
from databricks.sdk import WorkspaceClient
from utils import (
    get_catalogs, get_schemas, clear_current_step_messages, 
    upload_csv_to_table, upload_file_to_volume, list_s3_objects,
    copy_s3_to_databricks_volume, execute_copy_into_command
)

def render_data_upload_tab(workspace_url, token):
    """Render the Data Upload tab"""
    st.subheader("Data Upload")
    
    if not (workspace_url and token):
        st.warning("⚠️ Please configure Databricks connection first")
    else:
        try:
            client = WorkspaceClient(host=workspace_url, token=token)
            
            # Get catalogs and schemas  
            col1, col2 = st.columns(2)
            with col1:
                available_catalogs = get_catalogs(client)
                selected_catalog = st.selectbox("Select Catalog", options=available_catalogs, key="upload_catalog")
            
            with col2:
                if selected_catalog:
                    available_schemas = get_schemas(client, selected_catalog)
                    selected_schema = st.selectbox("Select Schema", options=available_schemas, key="upload_schema")
                else:
                    selected_schema = None
                    st.selectbox("Select Schema", options=[], disabled=True, key="upload_schema_disabled")
            
            if selected_catalog and selected_schema:
                st.divider()
                
                # File upload options
                upload_option = st.radio(
                    "Upload Source",
                    ["Local File", "S3 Bucket"],
                    horizontal=True
                )
                
                if upload_option == "Local File":
                    _render_local_file_upload(client, selected_catalog, selected_schema)
                
                elif upload_option == "S3 Bucket":
                    _render_s3_upload(client, selected_catalog, selected_schema)
                
                # Show uploaded files for selected catalog/schema
                st.divider()
                catalog_files = [f for f in st.session_state.uploaded_files 
                               if f.get('catalog') == selected_catalog and f.get('schema') == selected_schema]
                
                if catalog_files:
                    st.subheader(f"📁 Files in {selected_catalog}.{selected_schema}")
                    for file_info in catalog_files:
                        with st.expander(f"📄 {file_info['name']} ({file_info.get('source', 'Local')})"):
                            st.write(f"**Type:** {file_info['type']}")
                            st.write(f"**Source:** {file_info.get('source', 'Local Upload')}")
                            if 'table' in file_info:
                                st.write(f"**Table:** `{file_info['table']}`")
                            if 'path' in file_info:
                                st.write(f"**Path:** `{file_info['path']}`")
        
        except Exception as e:
            st.error(f"Error loading catalogs/schemas: {str(e)}")

def _render_local_file_upload(client, selected_catalog, selected_schema):
    """Render local file upload section"""
    # Multiple file upload
    uploaded_files = st.file_uploader(
        "Choose files", 
        type=['csv', 'pdf', 'txt', 'docx'],
        help="Supported formats: CSV, PDF, TXT, DOCX",
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.write(f"**Selected {len(uploaded_files)} file(s):**")
        
        # Group files by type
        csv_files = [f for f in uploaded_files if f.type == "text/csv"]
        document_files = [f for f in uploaded_files if f.type != "text/csv"]
        
        # Handle CSV files
        if csv_files:
            st.write("**CSV Files (will be uploaded as tables):**")
            col1, col2 = st.columns([3, 1])
            
            with col1:
                for csv_file in csv_files:
                    st.write(f"• {csv_file.name}")
            
            with col2:
                if st.button("📊 Upload CSV Files as Tables"):
                    clear_current_step_messages()
                    for csv_file in csv_files:
                        table_name = csv_file.name.replace('.csv', '').replace(' ', '_').lower()
                        if upload_csv_to_table(client, csv_file, selected_catalog, selected_schema, table_name):
                            st.session_state.uploaded_files.append({
                                'name': csv_file.name,
                                'type': 'CSV',
                                'table': f"{selected_catalog}.{selected_schema}.{table_name}",
                                'catalog': selected_catalog,
                                'schema': selected_schema
                            })
        
        # Handle document files
        if document_files:
            st.write("**Document Files (will be uploaded to volume):**")
            col1, col2 = st.columns([3, 1])
            
            with col1:
                for doc_file in document_files:
                    st.write(f"• {doc_file.name}")
                volume_name = st.text_input("Volume Name", value="document_storage")
            
            with col2:
                if st.button("📁 Upload Documents to Volume"):
                    clear_current_step_messages()
                    for doc_file in document_files:
                        file_path = upload_file_to_volume(
                            client, doc_file, selected_catalog, selected_schema, volume_name, doc_file.type
                        )
                        if file_path:
                            st.session_state.uploaded_files.append({
                                'name': doc_file.name,
                                'type': doc_file.type.split('/')[-1].upper(),
                                'path': file_path,
                                'catalog': selected_catalog,
                                'schema': selected_schema
                            })

def _render_s3_upload(client, selected_catalog, selected_schema):
    """Render S3 upload section"""
    st.subheader("📦 S3 Bucket Integration")
    
    with st.form("s3_form"):
        st.write("**AWS Credentials & S3 Configuration:**")
        
        col1, col2 = st.columns(2)
        with col1:
            aws_access_key = st.text_input(
                "AWS Access Key ID",
                type="password",
                help="Your AWS Access Key ID"
            )
            bucket_name = st.text_input(
                "S3 Bucket Name",
                placeholder="my-data-bucket",
                help="Name of your S3 bucket"
            )
            s3_prefix = st.text_input(
                "S3 Prefix/Path (Optional)",
                placeholder="data/files/",
                help="Optional prefix to filter objects (like a folder path)"
            )
        
        with col2:
            aws_secret_key = st.text_input(
                "AWS Secret Access Key",
                type="password",
                help="Your AWS Secret Access Key"
            )
            aws_region = st.selectbox(
                "AWS Region",
                options=[
                    "us-east-1", "us-east-2", "us-west-1", "us-west-2",
                    "eu-west-1", "eu-west-2", "eu-central-1", "ap-south-1",
                    "ap-southeast-1", "ap-southeast-2", "ap-northeast-1"
                ],
                index=0
            )
            volume_name = st.text_input(
                "Databricks Volume Name",
                value="s3_data_volume",
                help="Name for the volume where S3 data will be copied"
            )
        
        # Copy method selection
        st.write("**Copy Method:**")
        copy_method = st.radio(
            "Choose copy method",
            ["Direct Copy to Volume", "COPY INTO Command (for structured data)"],
            help="Direct copy downloads files to volume. COPY INTO is for CSV/JSON to tables."
        )
        
        list_objects_btn = st.form_submit_button("📋 List S3 Objects", use_container_width=True)
    
    # List S3 objects if credentials provided
    if list_objects_btn and aws_access_key and aws_secret_key and bucket_name:
        clear_current_step_messages()
        
        with st.spinner("Connecting to S3 and listing objects..."):
            s3_objects = list_s3_objects(
                aws_access_key, aws_secret_key, bucket_name, s3_prefix, aws_region
            )
        
        if s3_objects:
            st.success(f"✅ Found {len(s3_objects)} objects in S3 bucket")
            
            # Store S3 objects in session state for this catalog/schema
            s3_key = f"s3_objects_{selected_catalog}_{selected_schema}"
            st.session_state[s3_key] = {
                'objects': s3_objects,
                'credentials': {
                    'access_key': aws_access_key,
                    'secret_key': aws_secret_key,
                    'bucket': bucket_name,
                    'region': aws_region,
                    'volume': volume_name,
                    'method': copy_method
                }
            }
        else:
            st.error("❌ No objects found or error connecting to S3")
    
    # Display S3 objects if available
    s3_key = f"s3_objects_{selected_catalog}_{selected_schema}"
    if s3_key in st.session_state and st.session_state[s3_key]['objects']:
        st.divider()
        st.write("**S3 Objects Available for Copy:**")
        
        s3_data = st.session_state[s3_key]
        s3_objects = s3_data['objects']
        s3_creds = s3_data['credentials']
        
        # File selection interface
        selected_s3_objects = []
        
        # Select all checkbox
        select_all = st.checkbox("Select All Files")
        
        # Individual file selection
        for i, obj in enumerate(s3_objects):
            file_size_mb = obj['size'] / (1024 * 1024)
            checkbox_key = f"s3_select_{i}_{obj['key']}"
            
            is_selected = select_all or st.checkbox(
                f"📄 {obj['filename']} ({file_size_mb:.2f} MB) - {obj['key']}",
                key=checkbox_key,
                value=select_all
            )
            
            if is_selected:
                selected_s3_objects.append(obj)
        
        if selected_s3_objects:
            st.write(f"**Selected {len(selected_s3_objects)} file(s) for copy**")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Copy Method:** {s3_creds['method']}")
                st.write(f"**Target Volume:** `{selected_catalog}.{selected_schema}.{s3_creds['volume']}`")
            
            with col2:
                if st.button("📥 Copy from S3", use_container_width=True):
                    clear_current_step_messages()
                    
                    progress_bar = st.progress(0)
                    status_container = st.empty()
                    
                    if s3_creds['method'] == "Direct Copy to Volume":
                        # Direct copy method
                        status_container.text("Copying files from S3 to Databricks volume...")
                        
                        total_files = len(selected_s3_objects)
                        successful_copies = []
                        
                        for idx, obj in enumerate(selected_s3_objects):
                            progress = (idx + 1) / total_files
                            progress_bar.progress(progress)
                            
                            status_container.text(f"Copying {obj['filename']} ({idx + 1}/{total_files})")
                            
                            file_path = copy_s3_to_databricks_volume(
                                client, s3_creds['access_key'], s3_creds['secret_key'],
                                s3_creds['bucket'], obj['key'], selected_catalog, 
                                selected_schema, s3_creds['volume'], s3_creds['region']
                            )
                            
                            if file_path:
                                successful_copies.append(file_path)
                                
                                # Add to uploaded files
                                file_extension = obj['filename'].split('.')[-1].upper()
                                st.session_state.uploaded_files.append({
                                    'name': obj['filename'],
                                    'type': file_extension,
                                    'path': file_path,
                                    'catalog': selected_catalog,
                                    'schema': selected_schema,
                                    'source': 'S3'
                                })
                        
                        progress_bar.progress(1.0)
                        status_container.success(f"✅ Successfully copied {len(successful_copies)}/{total_files} files from S3")
                    
                    else:  # COPY INTO method
                        status_container.text("Executing COPY INTO commands...")
                        
                        for idx, obj in enumerate(selected_s3_objects):
                            progress = (idx + 1) / len(selected_s3_objects)
                            progress_bar.progress(progress)
                            
                            s3_path = f"s3://{s3_creds['bucket']}/{obj['key']}"
                            table_name = obj['filename'].replace('.', '_').lower()
                            file_format = "CSV" if obj['filename'].endswith('.csv') else "JSON"
                            
                            if execute_copy_into_command(
                                client, s3_path, table_name, selected_catalog, 
                                selected_schema, s3_creds['access_key'], 
                                s3_creds['secret_key'], file_format
                            ):
                                st.session_state.uploaded_files.append({
                                    'name': obj['filename'],
                                    'type': 'TABLE',
                                    'table': f"{selected_catalog}.{selected_schema}.{table_name}",
                                    'catalog': selected_catalog,
                                    'schema': selected_schema,
                                    'source': 'S3'
                                })
                        
                        progress_bar.progress(1.0)
                        status_container.success("✅ COPY INTO commands executed")
        
        # Clear S3 data button
        if st.button("🗑️ Clear S3 Data"):
            if s3_key in st.session_state:
                del st.session_state[s3_key]
            st.rerun()
