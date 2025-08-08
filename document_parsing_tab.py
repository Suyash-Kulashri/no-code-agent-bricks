import streamlit as st
import os
import pandas as pd
from document_parsing_utils import (
    get_sdk_client, get_catalogs_sdk, get_schemas_sdk, get_volumes_sdk,
    get_files_from_volume_sdk, parse_documents_to_table, get_spark_session
)

workspace_url = os.getenv("DATABRICKS_HOST")
token = os.getenv("DATABRICKS_TOKEN")
cluster_id = os.getenv("DATABRICKS_CLUSTER_ID")

if workspace_url and token:
    st.success('✅ Credentials provided')
else:
    st.warning('⚠️ Please provide credentials')

def render_document_parsing_tab():
    """Renders the professional UI using a Databricks SDK-first hybrid approach."""

    # Main content area
    if not all([workspace_url, token]):
        st.markdown("""
        <div class="info-box">
            <h3>🚀 Getting Started</h3>
            <p>Please provide your Databricks credentials in the sidebar to begin:</p>
            <ul>
                <li>Workspace URL</li>
                <li>Personal Access Token</li>
                <li>Cluster ID (for writing operations)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        return

    # Establish SDK client connection
    with st.spinner("🔄 Establishing connection to Databricks..."):
        client = get_sdk_client(host=workspace_url, token=token)
        
    if not client:
        st.error("❌ Could not establish SDK connection. Please verify your credentials.")
        return
    
    # Connection success
    st.markdown('<div class="success-box">✅ Connected to Databricks successfully!</div>', unsafe_allow_html=True)
    
    # --- Step 1: Data Source Selection ---
    st.markdown("""
    <div class="step-header">
        <h3>🎯 Step 1: Select Data Source Location</h3>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("📊 Loading available catalogs..."):
        available_catalogs = get_catalogs_sdk(client)
        
    if not available_catalogs:
        st.error("❌ No catalogs found. Please check your permissions.")
        return
    
    # Three-column layout for source selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_catalog = st.selectbox(
            "📁 Catalog", 
            options=[""] + available_catalogs,
            format_func=lambda x: "Select catalog..." if x == "" else x
        )
    
    with col2:
        if selected_catalog:
            with st.spinner("Loading schemas..."):
                available_schemas = get_schemas_sdk(client, selected_catalog)
            selected_schema = st.selectbox(
                "📂 Schema", 
                options=[""] + available_schemas,
                format_func=lambda x: "Select schema..." if x == "" else x
            )
        else:
            selected_schema = st.selectbox("📂 Schema", options=[""], disabled=True)
    
    with col3:
        if selected_catalog and selected_schema:
            with st.spinner("Loading volumes..."):
                available_volumes = get_volumes_sdk(client, selected_catalog, selected_schema)
            selected_volume = st.selectbox(
                "📦 Volume", 
                options=[""] + available_volumes,
                format_func=lambda x: "Select volume..." if x == "" else x
            )
        else:
            selected_volume = st.selectbox("📦 Volume", options=[""], disabled=True)

    # Show selected path
    if all([selected_catalog, selected_schema, selected_volume]):
        st.code(f"📍 Selected Path: /Volumes/{selected_catalog}/{selected_schema}/{selected_volume}")

    if not all([selected_catalog, selected_schema, selected_volume]):
        st.info("👆 Please complete your selection above to proceed to file selection.")
        return

    # --- Step 2: File Selection ---
    st.markdown("""
    <div class="step-header">
        <h3>📁 Step 2: Select Documents to Parse</h3>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("🔍 Scanning volume for supported files..."):
        files_in_volume = get_files_from_volume_sdk(client, selected_catalog, selected_schema, selected_volume)

    if not files_in_volume:
        st.markdown("""
        <div class="warning-box">
            <h4>📂 No Supported Files Found</h4>
            <p>No PDF or TXT files were found in the selected volume.</p>
            <p><strong>Supported formats:</strong> PDF, TXT</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # File statistics
    total_size = sum(f['size'] for f in files_in_volume) / (1024 * 1024)  # Convert to MB
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{len(files_in_volume)}</h2>
            <p>Files Available</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{total_size:.1f} MB</h2>
            <p>Total Size</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        pdf_count = sum(1 for f in files_in_volume if f['type'] == 'PDF')
        txt_count = len(files_in_volume) - pdf_count
        st.markdown(f"""
        <div class="metric-card">
            <h2>{pdf_count}📄 {txt_count}📝</h2>
            <p>PDF / TXT Files</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### 📋 File Selection")
    
    # Selection controls
    col1, col2 = st.columns([1, 3])
    with col1:
        select_all = st.checkbox("🔄 Select All Files", key="select_all_files")
    with col2:
        if select_all:
            st.info(f"✅ All {len(files_in_volume)} files selected")
    
    # File list with improved styling
    selected_files = []
    
    for i, file_info in enumerate(files_in_volume):
        size_mb = file_info['size'] / (1024 * 1024)
        
        # Create a more detailed file display
        col1, col2 = st.columns([0.1, 0.9])
        
        with col1:
            is_selected = st.checkbox(
                "", 
                value=select_all, 
                key=file_info['path'],
                label_visibility="collapsed"
            )
        
        with col2:
            file_icon = "📄" if file_info['type'] == 'PDF' else "📝"
            st.markdown(f"""
            <div class="file-item">
                <strong>{file_icon} {file_info['name']}</strong><br>
                <small>📏 Size: {size_mb:.2f} MB | 📂 Type: {file_info['type']}</small>
            </div>
            """, unsafe_allow_html=True)
        
        if is_selected:
            selected_files.append(file_info)

    if not selected_files:
        st.info("👆 Please select one or more files to proceed with parsing.")
        return
    
    # Selection summary
    selected_size = sum(f['size'] for f in selected_files) / (1024 * 1024)
    st.markdown(f"""
    <div class="success-box">
        ✅ <strong>{len(selected_files)} files selected</strong> ({selected_size:.1f} MB total)
    </div>
    """, unsafe_allow_html=True)
    
    # --- Step 3: Configuration ---
    st.markdown("""
    <div class="step-header">
        <h3>⚙️ Step 3: Configure Parsing Parameters</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Output Configuration")
        target_table_name = st.text_input(
            "📋 Target Table Name", 
            value=f"{selected_volume}_parsed",
            help="Name for the table that will store parsed documents"
        )
        
        if target_table_name:
            full_table_path = f"{selected_catalog}.{selected_schema}.{target_table_name}"
            st.code(f"📍 Full Table Path: {full_table_path}")
    
    with col2:
        st.markdown("#### 🔧 Chunking Parameters")
        parsing_config = {}
        
        parsing_config['chunk_size'] = st.slider(
            "📏 Chunk Size (characters)", 
            min_value=200, 
            max_value=8000, 
            value=2000, 
            step=100,
            help="Size of each text chunk in characters"
        )
        
        parsing_config['overlap'] = st.slider(
            "🔄 Chunk Overlap (characters)", 
            min_value=0, 
            max_value=1000, 
            value=200, 
            step=50,
            help="Number of overlapping characters between chunks"
        )

    # Configuration preview
    estimated_chunks = sum(
        max(1, (f['size'] // parsing_config['chunk_size'])) for f in selected_files
    )
    
    st.markdown(f"""
    <div class="info-box">
        <h4>📊 Processing Estimate</h4>
        <p><strong>Estimated chunks:</strong> ~{estimated_chunks:,}</p>
        <p><strong>Configuration:</strong> {parsing_config['chunk_size']} chars per chunk with {parsing_config['overlap']} char overlap</p>
    </div>
    """, unsafe_allow_html=True)
    
    # --- Step 4: Execute Parsing ---
    st.markdown("""
    <div class="step-header">
        <h3>🚀 Step 4: Execute Document Parsing</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Pre-execution checks
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ✅ Pre-execution Checklist")
        checks = [
            ("📁 Source selected", bool(selected_volume)),
            ("📄 Files selected", bool(selected_files)),
            ("📋 Table name provided", bool(target_table_name)),
            ("⚡ Cluster ID provided", bool(cluster_id))
        ]
        
        all_checks_passed = True
        for check_name, check_passed in checks:
            status = "✅" if check_passed else "❌"
            st.markdown(f"{status} {check_name}")
            if not check_passed:
                all_checks_passed = False
    
    with col2:
        if all_checks_passed:
            st.markdown("""
            <div class="success-box">
                <h4>🎯 Ready to Process</h4>
                <p>All requirements met. Click the button below to start parsing.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-box">
                <h4>⚠️ Requirements Missing</h4>
                <p>Please complete all requirements above.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Execute button
    if st.button(
        "🚀 Start Document Parsing", 
        type="primary", 
        disabled=(not all_checks_passed),
        use_container_width=True
    ):
        # Spark connection
        with st.spinner("⚡ Connecting to Spark cluster..."):
            spark = get_spark_session(host=workspace_url, token=token, cluster_id=cluster_id)
        
        if not spark:
            st.error("❌ Could not connect to Spark. Please verify your cluster ID and ensure the cluster is running.")
            return

        st.success("✅ Spark connection established successfully!")
        
        # Progress tracking
        progress_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            progress_text = st.empty()
            status_text = st.empty()
        
        def progress_callback(progress, message):
            progress_bar.progress(progress)
            progress_text.text(message)
            
        # Execute parsing
        with st.spinner("🔄 Processing documents..."):
            success = parse_documents_to_table(
                client, spark,
                selected_catalog, selected_schema, target_table_name,
                selected_files, parsing_config, progress_callback
            )
        
        # Results
        if success:
            st.balloons()
            st.markdown(f"""
            <div class="success-box">
                <h3>🎉 Processing Complete!</h3>
                <p><strong>Table created:</strong> <code>{selected_catalog}.{selected_schema}.{target_table_name}</code></p>
                <p><strong>Files processed:</strong> {len(selected_files)}</p>
                <p><strong>Ready for:</strong> Vector embeddings, search indexing, or analysis</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-box">
                <h3>⚠️ Processing Issues</h3>
                <p>Some issues occurred during processing. Please check the error messages above and try again.</p>
            </div>
            """, unsafe_allow_html=True)