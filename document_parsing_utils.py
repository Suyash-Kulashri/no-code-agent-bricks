import streamlit as st
import os
import fitz  # PyMuPDF library
import io
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.connect import DatabricksSession
from pyspark.sql import SparkSession
import traceback

# --- SDK Client-based Functions (for listing and reading) ---

def get_sdk_client(host, token):
    """Initializes and returns a Databricks SDK WorkspaceClient."""
    try:
        client = WorkspaceClient(host=host, token=token)
        list(client.catalogs.list())
        return client
    except Exception as e:
        st.error(f"❌ SDK connection failed: {e}")
        return None

def get_catalogs_sdk(client: WorkspaceClient):
    """Get all catalogs using the SDK."""
    try:
        return [c.name for c in client.catalogs.list()]
    except Exception as e:
        st.error(f"❌ Error fetching catalogs: {str(e)}")
        return []

def get_schemas_sdk(client: WorkspaceClient, catalog_name: str):
    """Get all schemas in a catalog using the SDK."""
    try:
        return [s.name for s in client.schemas.list(catalog_name=catalog_name)]
    except Exception as e:
        st.error(f"❌ Error fetching schemas: {str(e)}")
        return []

def get_volumes_sdk(client: WorkspaceClient, catalog_name: str, schema_name: str):
    """Get all volumes from a specific catalog and schema using the SDK."""
    try:
        volumes = list(client.volumes.list(catalog_name=catalog_name, schema_name=schema_name))
        return [volume.name for volume in volumes]
    except Exception as e:
        st.error(f"❌ Error fetching volumes: {str(e)}")
        return []

def get_files_from_volume_sdk(client: WorkspaceClient, catalog_name: str, schema_name: str, volume_name: str):
    """Get all files from a specific volume using the SDK."""
    try:
        volume_path = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}"
        files_response = client.files.list_directory_contents(directory_path=volume_path)
        
        files = []
        supported_extensions = ['pdf', 'txt']
        for file_info in files_response:
            if not file_info.is_directory:
                file_ext = file_info.name.split('.')[-1].lower() if '.' in file_info.name else 'unknown'
                if file_ext in supported_extensions:
                    file_type = 'PDF' if file_ext == 'pdf' else 'Text'
                    files.append({
                        'name': file_info.name,
                        'path': file_info.path,
                        'size': file_info.file_size,
                        'type': file_type,
                    })
        return files
    except Exception as e:
        st.error(f"❌ Error fetching files from volume {volume_name}: {str(e)}")
        return []

# --- MODIFIED FUNCTION ---
def download_file_content_sdk(client: WorkspaceClient, file_path: str):
    """Downloads a file's content and returns the raw BYTES."""
    try:
        response = client.files.download(file_path)
        # --- THE DEFINITIVE FIX ---
        # The response object has a .contents attribute which is a stream.
        # We must call .read() on the .contents stream to get the binary data.
        return response.contents.read()
    except Exception as e:
        st.error(f"❌ Error downloading {os.path.basename(file_path)}: {e}")
        return None

# --- Spark Connect-based Functions (for writing tables) ---

def get_spark_session(host, token, cluster_id):
    """Initializes and returns a Spark session."""
    try:
        return DatabricksSession.builder.remote(host=host, token=token, cluster_id=cluster_id).getOrCreate()
    except Exception as e:
        st.error(f"❌ Failed to create Spark Session for writing data: {e}")
        return None

# --- Parsing and Chunking Logic ---

def chunk_text(text, chunk_size, overlap):
    """Splits text into overlapping chunks."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def parse_documents_to_table(client: WorkspaceClient, spark: SparkSession, catalog, schema, table_name, selected_files, parsing_config, progress_callback):
    """Downloads files via SDK, parses them, and saves to a table via Spark."""
    try:
        progress_callback(0.1, "🚀 Starting document processing...")
        parsed_data = []
        
        for i, file_info in enumerate(selected_files):
            progress_callback(0.1 + (i / len(selected_files)) * 0.8, f"📄 Processing {file_info['name']}...")
            
            # This now correctly receives the raw bytes of the file.
            content_bytes = download_file_content_sdk(client, file_info['path'])
            if not content_bytes:
                st.warning(f"⚠️ Could not download {file_info['name']}. Skipping.")
                continue

            full_text = ""
            try:
                if file_info['type'] == 'PDF':
                    # PyMuPDF's fitz.open can now correctly process the bytes.
                    with fitz.open(stream=content_bytes, filetype="pdf") as doc:
                        for page in doc:
                            full_text += page.get_text() + "\n"
                
                elif file_info['type'] == 'Text':
                    # We can now directly decode the bytes.
                    full_text = content_bytes.decode('utf-8', errors='ignore')

            except Exception as parsing_error:
                st.warning(f"⚠️ Failed to parse {file_info['name']}. Error: {parsing_error}. Skipping.")
                continue
            
            if not full_text.strip():
                st.warning(f"⚠️ No text could be extracted from {file_info['name']}. It might be an image-only PDF. Skipping.")
                continue

            text_chunks = chunk_text(full_text, parsing_config['chunk_size'], parsing_config['overlap'])
            
            for chunk_idx, chunk_content in enumerate(text_chunks):
                if not chunk_content.strip():
                    continue
                
                record = {
                    'document_name': file_info['name'],
                    'document_path': file_info['path'],
                    'chunk_index': chunk_idx,
                    'content': chunk_content,
                    'word_count': len(chunk_content.split()),
                    'char_count': len(chunk_content)
                }
                parsed_data.append(record)
        
        if not parsed_data:
            st.error("❌ Parsing completed, but no text was extracted from any of the selected files.")
            return False
            
        progress_callback(0.9, f"💾 Creating Spark DataFrame and saving to table...")
        
        df = spark.createDataFrame(parsed_data)
        
        # Add primary key as auto-incremented row number
        from pyspark.sql.window import Window
        from pyspark.sql.functions import row_number
        
        window_spec = Window.orderBy(df.columns[0])  # Order by first column for consistent numbering
        df_with_pk = df.withColumn("id", row_number().over(window_spec))
        
        # Reorder columns to have 'id' as the first column
        columns = ["id"] + [col for col in df.columns]
        df_final = df_with_pk.select(columns)
        
        full_table_name = f"`{catalog}`.`{schema}`.`{table_name}`"
        df_final.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(full_table_name)
        
        st.success(f"🎉 Parsed {len(selected_files)} files into {len(parsed_data)} chunks!")
        return True
    
    except Exception as e:
        st.error(f"❌ Error during parsing job: {e}")
        traceback.print_exc()
        return False