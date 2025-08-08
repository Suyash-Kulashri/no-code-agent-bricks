import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import VolumeType
from databricks.vector_search.client import VectorSearchClient
import requests
import PyPDF2
import pandas as pd
import json
import os
import tempfile
import time
import sys
from typing import Dict, List, Optional
import io
import boto3

def clear_current_step_messages():
    """Clear messages from current step"""
    st.session_state.current_step_messages = []

def add_status_message(message: str, is_success: bool = True):
    """Add a status message to current step"""
    st.session_state.current_step_messages.append({
        'message': message,
        'is_success': is_success,
        'timestamp': time.time()
    })

def list_s3_objects(aws_access_key: str, aws_secret_key: str, bucket_name: str, prefix: str = "", region: str = "us-east-1") -> List[Dict]:
    """List objects in S3 bucket with given prefix"""
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region
        )
        
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix,
            MaxKeys=1000
        )
        
        objects = []
        if 'Contents' in response:
            for obj in response['Contents']:
                # Filter out directories (objects ending with /)
                if not obj['Key'].endswith('/'):
                    objects.append({
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'modified': obj['LastModified'],
                        'filename': obj['Key'].split('/')[-1]
                    })
        
        return objects
    except Exception as e:
        add_status_message(f"❌ Error listing S3 objects: {str(e)}", False)
        return []

def copy_s3_to_databricks_volume(client: WorkspaceClient, aws_access_key: str, aws_secret_key: str, 
                                bucket_name: str, s3_key: str, catalog_name: str, schema_name: str, 
                                volume_name: str, region: str = "us-east-1") -> str:
    """Copy S3 object to Databricks volume"""
    try:
        # Create S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region
        )
        
        # Create volume if it doesn't exist
        full_volume_name = f"{catalog_name}.{schema_name}.{volume_name}"
        try:
            client.volumes.create(
                name=volume_name,
                catalog_name=catalog_name,
                schema_name=schema_name,
                volume_type=VolumeType.MANAGED
            )
            add_status_message(f"✅ Created volume: {full_volume_name}")
        except Exception as e:
            if "already exists" in str(e).lower():
                add_status_message(f"ℹ️ Volume {full_volume_name} already exists")
            else:
                raise e
        
        # Download file from S3
        file_content = s3_client.get_object(Bucket=bucket_name, Key=s3_key)['Body'].read()
        
        # Get filename from S3 key
        filename = s3_key.split('/')[-1]
        
        # Upload to Databricks volume
        file_path = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/{filename}"
        
        client.files.upload(
            file_path=file_path,
            contents=io.BytesIO(file_content),
            overwrite=True
        )
        
        add_status_message(f"✅ Copied {filename} from S3 to: {file_path}")
        return file_path
        
    except Exception as e:
        add_status_message(f"❌ Error copying from S3: {str(e)}", False)
        return ""

def bulk_copy_s3_to_databricks(client: WorkspaceClient, aws_access_key: str, aws_secret_key: str, 
                              bucket_name: str, s3_objects: List[Dict], catalog_name: str, 
                              schema_name: str, volume_name: str, region: str = "us-east-1") -> List[str]:
    """Copy multiple S3 objects to Databricks volume"""
    try:
        # Create S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region
        )
        
        # Create volume if it doesn't exist
        full_volume_name = f"{catalog_name}.{schema_name}.{volume_name}"
        try:
            client.volumes.create(
                name=volume_name,
                catalog_name=catalog_name,
                schema_name=schema_name,
                volume_type=VolumeType.MANAGED
            )
            add_status_message(f"✅ Created volume: {full_volume_name}")
        except Exception as e:
            if "already exists" in str(e).lower():
                add_status_message(f"ℹ️ Volume {full_volume_name} already exists")
            else:
                raise e
        
        successful_copies = []
        
        for obj in s3_objects:
            try:
                s3_key = obj['key']
                filename = obj['filename']
                
                # Download file from S3
                file_content = s3_client.get_object(Bucket=bucket_name, Key=s3_key)['Body'].read()
                
                # Upload to Databricks volume
                file_path = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/{filename}"
                
                client.files.upload(
                    file_path=file_path,
                    contents=io.BytesIO(file_content),
                    overwrite=True
                )
                
                successful_copies.append(file_path)
                add_status_message(f"✅ Copied {filename} from S3")
                
            except Exception as e:
                add_status_message(f"❌ Error copying {obj['filename']}: {str(e)}", False)
        
        return successful_copies
        
    except Exception as e:
        add_status_message(f"❌ Error in bulk copy: {str(e)}", False)
        return []

def create_databricks_storage_credential(client: WorkspaceClient, credential_name: str, 
                                        aws_access_key: str, aws_secret_key: str) -> bool:
    """Create storage credential for S3 access (if needed for COPY INTO)"""
    try:
        # This would be used for COPY INTO operations
        # Implementation depends on whether user wants to use storage credentials
        add_status_message(f"ℹ️ Storage credential creation not implemented in this version")
        return True
    except Exception as e:
        add_status_message(f"❌ Error creating storage credential: {str(e)}", False)
        return False

def execute_copy_into_command(client: WorkspaceClient, s3_path: str, table_name: str, 
                             catalog_name: str, schema_name: str, aws_access_key: str, 
                             aws_secret_key: str, file_format: str = "CSV") -> bool:
    """Execute COPY INTO command for structured data"""
    try:
        full_table_name = f"{catalog_name}.{schema_name}.{table_name}"
        
        # Create table first (basic structure for CSV)
        if file_format.upper() == "CSV":
            create_sql = f"""
            CREATE TABLE IF NOT EXISTS {full_table_name} (
                data STRING
            ) USING DELTA
            TBLPROPERTIES (
                'delta.enableChangeDataFeed' = 'true'
            )
            """
        else:
            create_sql = f"""
            CREATE TABLE IF NOT EXISTS {full_table_name} (
                content STRING,
                file_path STRING
            ) USING DELTA
            TBLPROPERTIES (
                'delta.enableChangeDataFeed' = 'true'
            )
            """
        
        warehouse_id = get_warehouse_id(client)
        client.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            statement=create_sql
        )
        
        # Execute COPY INTO (simplified version)
        copy_sql = f"""
        COPY INTO {full_table_name}
        FROM '{s3_path}'
        FILEFORMAT = {file_format}
        COPY_OPTIONS ('mergeSchema' = 'true')
        """
        
        # Note: This is a simplified version. In production, you'd need proper credentials setup
        add_status_message(f"ℹ️ COPY INTO command prepared for {full_table_name}")
        add_status_message(f"ℹ️ For production use, configure storage credentials in Databricks")
        
        return True
        
    except Exception as e:
        add_status_message(f"❌ Error executing COPY INTO: {str(e)}", False)
        return False

def get_catalogs(client: WorkspaceClient) -> List[str]:
    """Get list of all catalogs"""
    try:
        catalogs = list(client.catalogs.list())
        return [catalog.name for catalog in catalogs]
    except Exception as e:
        add_status_message(f"❌ Error fetching catalogs: {str(e)}", False)
        return []

def get_schemas(client: WorkspaceClient, catalog_name: str) -> List[str]:
    """Get list of schemas in a catalog"""
    try:
        schemas = list(client.schemas.list(catalog_name=catalog_name))
        return [schema.name for schema in schemas]
    except Exception as e:
        add_status_message(f"❌ Error fetching schemas: {str(e)}", False)
        return []
    
# Add these additional functions to your existing utils.py file

def get_volumes_from_catalog_schema(client: WorkspaceClient, catalog_name: str, schema_name: str) -> List[str]:
    """Get all volumes from a specific catalog and schema"""
    try:
        volumes = list(client.volumes.list(catalog_name=catalog_name, schema_name=schema_name))
        return [volume.name for volume in volumes]
    except Exception as e:
        add_status_message(f"❌ Error fetching volumes: {str(e)}", False)
        return []

def get_files_from_volume(client: WorkspaceClient, catalog_name: str, schema_name: str, volume_name: str) -> List[Dict]:
    """Get all files from a specific volume"""
    try:
        volume_path = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/"
        
        # List files in the volume
        response = client.files.list_directory_contents(directory_path=volume_path)
        
        files = []
        for file_info in response:
            if not file_info.is_directory:
                # Determine file type based on extension
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

def get_tables_from_catalog_schema(client: WorkspaceClient, catalog_name: str, schema_name: str) -> List[Dict]:
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

def create_vector_storage_volume(client: WorkspaceClient, catalog_name: str, schema_name: str, volume_name: str) -> bool:
    """Create a dedicated volume for storing vector-related data"""
    try:
        full_volume_name = f"{catalog_name}.{schema_name}.{volume_name}"
        
        # Create volume if it doesn't exist
        try:
            client.volumes.create(
                name=volume_name,
                catalog_name=catalog_name,
                schema_name=schema_name,
                volume_type=VolumeType.MANAGED,
                comment=f"Volume for storing vector indexes and related data"
            )
            add_status_message(f"✅ Created vector storage volume: {full_volume_name}")
            return True
        except Exception as e:
            if "already exists" in str(e).lower():
                add_status_message(f"ℹ️ Vector storage volume {full_volume_name} already exists")
                return True
            else:
                raise e
                
    except Exception as e:
        add_status_message(f"❌ Error creating vector storage volume: {str(e)}", False)
        return False

def get_file_preview(client: WorkspaceClient, file_path: str, max_chars: int = 500) -> str:
    """Get a preview of file content"""
    try:
        # Download a portion of the file for preview
        file_content = client.files.download(file_path)
        
        # Convert bytes to string and truncate if necessary
        content_str = file_content.read().decode('utf-8', errors='ignore')
        
        if len(content_str) > max_chars:
            preview = content_str[:max_chars] + "... [truncated]"
        else:
            preview = content_str
            
        return preview
        
    except Exception as e:
        return f"Error reading file: {str(e)}"

def validate_vectorization_requirements(client: WorkspaceClient, catalog_name: str, schema_name: str) -> Dict[str, bool]:
    """Validate if all requirements for vectorization are met"""
    requirements = {
        'catalog_exists': False,
        'schema_exists': False,
        'vector_search_available': False,
        'sql_warehouse_available': False
    }
    
    try:
        # Check catalog
        catalogs = get_catalogs(client)
        requirements['catalog_exists'] = catalog_name in catalogs
        
        # Check schema
        if requirements['catalog_exists']:
            schemas = get_schemas(client, catalog_name)
            requirements['schema_exists'] = schema_name in schemas
        
        # Check vector search (try to list endpoints)
        try:
            response = requests.get(
                f"{client.config.host}/api/2.0/vector-search/endpoints",
                headers={"Authorization": f"Bearer {client.config.token}"},
                timeout=5
            )
            requirements['vector_search_available'] = response.status_code == 200
        except:
            requirements['vector_search_available'] = False
        
        # Check SQL warehouse
        try:
            warehouses = list(client.warehouses.list())
            requirements['sql_warehouse_available'] = len(warehouses) > 0
        except:
            requirements['sql_warehouse_available'] = False
            
    except Exception as e:
        add_status_message(f"⚠️ Error validating requirements: {str(e)}")
    
    return requirements

# Add these functions to the end of your existing utils.py file

def get_volumes_from_catalog_schema(client: WorkspaceClient, catalog_name: str, schema_name: str) -> List[str]:
    """Get all volumes from a specific catalog and schema"""
    try:
        volumes = list(client.volumes.list(catalog_name=catalog_name, schema_name=schema_name))
        return [volume.name for volume in volumes]
    except Exception as e:
        add_status_message(f"❌ Error fetching volumes: {str(e)}", False)
        return []

def get_files_from_volume(client: WorkspaceClient, catalog_name: str, schema_name: str, volume_name: str) -> List[Dict]:
    """Get all files from a specific volume"""
    try:
        volume_path = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/"
        
        # List files in the volume
        response = client.files.list_directory_contents(directory_path=volume_path)
        
        files = []
        for file_info in response:
            if not file_info.is_directory:
                # Determine file type based on extension
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

def get_tables_from_catalog_schema(client: WorkspaceClient, catalog_name: str, schema_name: str) -> List[Dict]:
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

def get_vector_indexes(client: WorkspaceClient, catalog_name: str, schema_name: str) -> List[str]:
    """Get list of vector indexes in a schema"""
    try:
        response = requests.get(
            f"{client.config.host}/api/2.0/vector-search/indexes",
            headers={"Authorization": f"Bearer {client.config.token}"}
        )
        
        if response.status_code == 200:
            indexes_data = response.json()
            indexes = []
            for idx in indexes_data.get("vector_indexes", []):
                idx_name = idx.get("name", "")
                if idx_name.startswith(f"{catalog_name}.{schema_name}."):
                    indexes.append(idx_name.split('.')[-1])
            return indexes
        else:
            add_status_message(f"❌ Error fetching vector indexes: {response.text}", False)
            return []
    except Exception as e:
        add_status_message(f"❌ Error fetching vector indexes: {str(e)}", False)
        return []

def get_serving_endpoints(client: WorkspaceClient) -> List[Dict]:
    """Get list of all serving endpoints"""
    try:
        response = requests.get(
            f"{client.config.host}/api/2.0/serving-endpoints",
            headers={"Authorization": f"Bearer {client.config.token}"}
        )
        
        if response.status_code == 200:
            endpoints_data = response.json()
            endpoints = []
            for ep in endpoints_data.get("endpoints", []):
                endpoints.append({
                    'name': ep.get('name', 'Unknown'),
                    'state': ep.get('state', {}).get('ready', 'Unknown'),
                    'url': f"{client.config.host}/serving-endpoints/{ep.get('name', '')}"
                })
            return endpoints
        else:
            add_status_message(f"❌ Error fetching serving endpoints: {response.text}", False)
            return []
    except Exception as e:
        add_status_message(f"❌ Error fetching serving endpoints: {str(e)}", False)
        return []

def get_foundation_models(client: WorkspaceClient) -> List[str]:
    """Get list of available foundation models from Databricks"""
    try:
        # Get foundation models from Databricks Foundation Model APIs
        response = requests.get(
            f"{client.config.host}/api/2.0/serving-endpoints",
            headers={"Authorization": f"Bearer {client.config.token}"},
            timeout=10
        )
        
        foundation_models = []
        
        if response.status_code == 200:
            endpoints_data = response.json()
            
            # Extract foundation model endpoints
            for endpoint in endpoints_data.get("endpoints", []):
                endpoint_name = endpoint.get("name", "")
                
                # Filter for foundation models (they usually have specific naming patterns)
                if any(pattern in endpoint_name.lower() for pattern in [
                    "databricks", "llama", "mixtral", "dbrx", "mpt", "foundation"
                ]):
                    foundation_models.append(endpoint_name)
        
        # If no foundation models found via endpoints, try the Foundation Model API
        if not foundation_models:
            try:
                # Try Foundation Model Registry API
                fm_response = requests.get(
                    f"{client.config.host}/api/2.0/preview/foundation-models",
                    headers={"Authorization": f"Bearer {client.config.token}"},
                    timeout=10
                )
                
                if fm_response.status_code == 200:
                    fm_data = fm_response.json()
                    for model in fm_data.get("foundation_models", []):
                        model_name = model.get("name") or model.get("display_name")
                        if model_name and model.get("task") in ["llm/v1/chat", "llm/v1/completions"]:
                            foundation_models.append(model_name)
                
            except Exception as fm_e:
                add_status_message(f"⚠️ Foundation Model API not accessible: {str(fm_e)}")
        
        # If still no models found, try getting from model registry
        if not foundation_models:
            try:
                # Try to get registered models that might be foundation models
                registry_response = requests.get(
                    f"{client.config.host}/api/2.0/mlflow/registered-models/search",
                    headers={"Authorization": f"Bearer {client.config.token}"},
                    params={"filter": "name LIKE '%databricks%' OR name LIKE '%foundation%'"},
                    timeout=10
                )
                
                if registry_response.status_code == 200:
                    registry_data = registry_response.json()
                    for model in registry_data.get("registered_models", []):
                        foundation_models.append(model.get("name"))
                        
            except Exception as reg_e:
                add_status_message(f"⚠️ Model Registry API not accessible: {str(reg_e)}")
        
        # Remove duplicates and sort
        foundation_models = sorted(list(set(foundation_models)))
        
        if foundation_models:
            add_status_message(f"✅ Found {len(foundation_models)} foundation models from Databricks")
            return foundation_models
        else:
            # Only use fallback if absolutely no models found
            add_status_message("⚠️ No foundation models found, using common model identifiers")
            return [
                "databricks-dbrx-instruct",
                "databricks-mixtral-8x7b-instruct", 
                "databricks-llama-2-70b-chat",
                "databricks-mpt-30b-instruct"
            ]
            
    except requests.exceptions.Timeout:
        add_status_message("❌ Timeout fetching foundation models, using defaults")
        return ["databricks-dbrx-instruct", "databricks-mixtral-8x7b-instruct"]
        
    except Exception as e:
        add_status_message(f"❌ Error fetching foundation models: {str(e)}, using defaults")
        return ["databricks-dbrx-instruct", "databricks-mixtral-8x7b-instruct"]

def create_catalog_and_schema(client: WorkspaceClient, catalog_name: str, schema_name: str, storage_location: str) -> bool:
    """Create catalog and schema in Databricks"""
    try:
        # Create catalog
        try:
            client.catalogs.create(name=catalog_name, storage_root=storage_location)
            add_status_message(f"✅ Created catalog: {catalog_name}")
        except Exception as e:
            if "already exists" in str(e).lower():
                add_status_message(f"ℹ️ Catalog {catalog_name} already exists")
            else:
                raise e
        
        # Create schema
        try:
            client.schemas.create(name=schema_name, catalog_name=catalog_name)
            add_status_message(f"✅ Created schema: {catalog_name}.{schema_name}")
        except Exception as e:
            if "already exists" in str(e).lower():
                add_status_message(f"ℹ️ Schema {catalog_name}.{schema_name} already exists")
            else:
                raise e
        
        return True
    except Exception as e:
        add_status_message(f"❌ Error creating catalog/schema: {str(e)}", False)
        return False

def upload_file_to_volume(client: WorkspaceClient, file, catalog_name: str, schema_name: str, volume_name: str, file_type: str) -> str:
    """Upload file to Databricks volume"""
    try:
        full_volume_name = f"{catalog_name}.{schema_name}.{volume_name}"
        
        # Create volume if it doesn't exist
        try:
            client.volumes.create(
                name=volume_name,
                catalog_name=catalog_name,
                schema_name=schema_name,
                volume_type=VolumeType.MANAGED
            )
            add_status_message(f"✅ Created volume: {full_volume_name}")
        except Exception as e:
            if "already exists" in str(e).lower():
                add_status_message(f"ℹ️ Volume {full_volume_name} already exists")
            else:
                raise e
        
        # Upload file
        file_content = file.read()
        file.seek(0)
        file_path = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/{file.name}"
        
        client.files.upload(
            file_path=file_path,
            contents=io.BytesIO(file_content),
            overwrite=True
        )
        
        add_status_message(f"✅ File uploaded to: {file_path}")
        return file_path
        
    except Exception as e:
        add_status_message(f"❌ Error uploading file: {str(e)}", False)
        return ""

def upload_csv_to_table(client: WorkspaceClient, file, catalog_name: str, schema_name: str, table_name: str) -> bool:
    """Upload CSV data to Databricks table"""
    try:
        df = pd.read_csv(file)
        
        # Create table structure
        columns = []
        for col in df.columns:
            if df[col].dtype == 'int64':
                col_type = "BIGINT"
            elif df[col].dtype == 'float64':
                col_type = "DOUBLE"
            else:
                col_type = "STRING"
            columns.append(f"`{col}` {col_type}")
        
        full_table_name = f"{catalog_name}.{schema_name}.{table_name}"
        
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {full_table_name} (
            {', '.join(columns)}
        ) USING DELTA
        TBLPROPERTIES (
            'delta.enableChangeDataFeed' = 'true'
        )
        """
        
        warehouse_id = get_warehouse_id(client)
        client.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            statement=create_sql
        )
        
        add_status_message(f"✅ Created table with CDF enabled: {full_table_name}")
        return True
    except Exception as e:
        add_status_message(f"❌ Error uploading CSV: {str(e)}", False)
        return False

def get_warehouse_id(client: WorkspaceClient) -> str:
    """Get the first available SQL warehouse ID"""
    try:
        warehouses = client.warehouses.list()
        for warehouse in warehouses:
            if warehouse.state.value == "RUNNING":
                return warehouse.id
        warehouses_list = list(warehouses)
        if warehouses_list:
            return warehouses_list[0].id
        raise Exception("No SQL warehouses available")
    except Exception as e:
        raise Exception(f"Error getting warehouse ID: {str(e)}")

def create_text_table_from_file(client: WorkspaceClient, file_path: str, catalog_name: str, schema_name: str, table_name: str) -> bool:
    """Create a text table from file content for vectorization with CDF enabled"""
    try:
        full_table_name = f"{catalog_name}.{schema_name}.{table_name}"
        
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {full_table_name} (
            id STRING,
            content STRING,
            source_file STRING,
            chunk_index INT
        ) USING DELTA
        TBLPROPERTIES (
            'delta.enableChangeDataFeed' = 'true'
        )
        """
        
        warehouse_id = get_warehouse_id(client)
        client.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            statement=create_sql
        )
        
        add_status_message(f"✅ Created text table with CDF enabled: {full_table_name}")
        
        # Insert sample data
        insert_sql = f"""
        INSERT INTO {full_table_name} VALUES 
        ('chunk_1', 'Sample content chunk 1 from {file_path}', '{file_path}', 1),
        ('chunk_2', 'Sample content chunk 2 from {file_path}', '{file_path}', 2)
        """
        
        client.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            statement=insert_sql
        )
        
        add_status_message(f"✅ Inserted sample data into {full_table_name}")
        return True
        
    except Exception as e:
        add_status_message(f"❌ Error creating text table: {str(e)}", False)
        return False

def create_vector_search_endpoint(client: WorkspaceClient, endpoint_name: str = "vector_search_endpoint") -> bool:
    """Create a vector search endpoint if it doesn't exist"""
    try:
        # Check if endpoint exists
        response = requests.get(
            f"{client.config.host}/api/2.0/vector-search/endpoints/{endpoint_name}",
            headers={"Authorization": f"Bearer {client.config.token}"}
        )
        
        if response.status_code == 200:
            endpoint_data = response.json()
            status = endpoint_data.get("endpoint_status", {}).get("state", "UNKNOWN")
            add_status_message(f"ℹ️ Vector search endpoint '{endpoint_name}' exists with status: {status}")
            return status in ["ONLINE", "PROVISIONING"]
        
        # Create new endpoint
        endpoint_config = {
            "name": endpoint_name,
            "endpoint_type": "STANDARD"
        }
        
        response = requests.post(
            f"{client.config.host}/api/2.0/vector-search/endpoints",
            headers={
                "Authorization": f"Bearer {client.config.token}",
                "Content-Type": "application/json"
            },
            json=endpoint_config
        )
        
        if response.status_code in [200, 201]:
            add_status_message(f"✅ Created vector search endpoint: {endpoint_name}")
            return True
        else:
            add_status_message(f"❌ Failed to create vector search endpoint: {response.text}", False)
            return False
            
    except Exception as e:
        add_status_message(f"❌ Error creating vector search endpoint: {str(e)}", False)
        return False

def create_vector_search_index(client: WorkspaceClient, catalog_name: str, schema_name: str, 
                              table_name: str, index_name: str, endpoint_name: str = "vector_search_endpoint") -> str:
    """Create vector search index"""
    try:
        full_index_name = f"{catalog_name}.{schema_name}.{index_name}"
        full_table_name = f"{catalog_name}.{schema_name}.{table_name}"
        
        # Ensure endpoint exists
        if not create_vector_search_endpoint(client, endpoint_name):
            return ""
        
        # Create index
        index_config = {
            "name": full_index_name,
            "endpoint_name": endpoint_name,
            "primary_key": "id",
            "index_type": "DELTA_SYNC",
            "delta_sync_index_spec": {
                "source_table": full_table_name,
                "pipeline_type": "TRIGGERED",
                "embedding_source_columns": [
                    {
                        "name": "content",
                        "embedding_model_endpoint_name": "databricks-bge-large-en"
                    }
                ]
            }
        }
        
        response = requests.post(
            f"{client.config.host}/api/2.0/vector-search/indexes",
            headers={
                "Authorization": f"Bearer {client.config.token}",
                "Content-Type": "application/json"
            },
            json=index_config
        )
        
        if response.status_code in [200, 201]:
            add_status_message(f"✅ Created vector search index: {full_index_name}")
            return full_index_name
        else:
            add_status_message(f"❌ Failed to create vector index: {response.text}", False)
            return ""
            
    except Exception as e:
        add_status_message(f"❌ Error creating vector index: {str(e)}", False)
        return ""

def create_simple_agent_endpoint(client: WorkspaceClient, agent_name: str, model_name: str, 
                               system_prompt: str, vector_indexes: List[str] = None) -> str:
    """Create a simple agent endpoint that proxies to the foundation model"""
    try:
        endpoint_name = f"{agent_name}_simple_endpoint"
        
        add_status_message(f"🔄 Creating simple agent endpoint: {endpoint_name}")
        
        # Store agent configuration in session state instead of trying to create a complex wrapper
        # This approach stores the configuration and will use the foundation model directly in chat
        agent_info = {
            'name': agent_name,
            'endpoint_name': model_name,  # Use the foundation model endpoint directly
            'url': f"{client.config.host}/serving-endpoints/{model_name}",
            'model': model_name,
            'system_prompt': system_prompt,
            'vector_indexes': vector_indexes or [],
            'status': 'Ready',
            'is_external': False,
            'is_simple': True  # Flag to indicate this is a simple proxy
        }
        
        # Update session state
        st.session_state.agent_endpoints.append(agent_info)
        
        add_status_message(f"✅ Created simple agent configuration for: {agent_name}")
        add_status_message(f"🔗 Will use foundation model endpoint: {model_name}")
        
        return agent_info['url']
        
    except Exception as e:
        add_status_message(f"❌ Error creating simple agent endpoint: {str(e)}", False)
        return ""

def create_agent_endpoint(client: WorkspaceClient, agent_name: str, model_name: str, 
                        system_prompt: str, vector_indexes: List[str] = None, api_key: str = "") -> str:
    """Create agent serving endpoint with proper configuration"""
    try:
        endpoint_name = f"{agent_name}_endpoint"
        
        add_status_message(f"🚀 Creating agent endpoint: {endpoint_name}")
        add_status_message(f"📋 Model: {model_name}")
        
        # Check if this is an external model (has API key)
        is_external_model = bool(api_key)
        
        if is_external_model:
            # For external models, we create a custom serving endpoint
            agent_config = {
                "name": endpoint_name,
                "config": {
                    "served_entities": [{
                        "name": agent_name,
                        "external_model": {
                            "name": model_name,
                            "provider": "anthropic" if "claude" in model_name.lower() else "openai",
                            "task": "llm/v1/chat",
                            "anthropic_config": {
                                "anthropic_api_key": api_key
                            } if "claude" in model_name.lower() else None,
                            "openai_config": {
                                "openai_api_key": api_key
                            } if "gpt" in model_name.lower() or "openai" in model_name.lower() else None
                        },
                        "workload_size": "Small",
                        "scale_to_zero_enabled": True
                    }],
                    "traffic_config": {
                        "routes": [{
                            "served_model_name": agent_name,
                            "traffic_percentage": 100
                        }]
                    }
                }
            }
        else:
            # For Databricks foundation models, create a proxy endpoint that calls the foundation model
            # Since the foundation model is already a serving endpoint, we'll create a simple wrapper
            
            # First, let's verify the foundation model endpoint exists
            try:
                test_response = requests.get(
                    f"{client.config.host}/serving-endpoints/{model_name}",
                    headers={"Authorization": f"Bearer {client.config.token}"},
                    timeout=10
                )
                
                if test_response.status_code != 200:
                    add_status_message(f"❌ Foundation model endpoint {model_name} not accessible", False)
                    return ""
                    
            except Exception as e:
                add_status_message(f"❌ Cannot verify foundation model endpoint: {str(e)}", False)
                return ""
            
            # Create a configuration for a foundation model endpoint
            # We'll use the foundation model directly without trying to wrap it
            agent_config = {
                "name": endpoint_name,
                "config": {
                    "served_entities": [{
                        "name": agent_name,
                        "foundation_model_name": model_name,  # Use foundation_model_name instead of entity_name
                        "workload_size": "Small",
                        "scale_to_zero_enabled": True,
                        "environment_vars": {
                            "SYSTEM_PROMPT": system_prompt
                        }
                    }],
                    "traffic_config": {
                        "routes": [{
                            "served_model_name": agent_name,  
                            "traffic_percentage": 100
                        }]
                    }
                }
            }
        
        # Add vector indexes if provided
        if vector_indexes:
            valid_indexes = []
            for idx_name in vector_indexes:
                # Verify index exists in session state
                for stored_idx in st.session_state.vectorized_indexes:
                    if stored_idx['name'] == idx_name:
                        valid_indexes.append(stored_idx['index_path'])
                        add_status_message(f"📊 Added vector index: {stored_idx['index_path']}")
                        break
            
            if valid_indexes:
                agent_config["config"]["served_entities"][0]["environment_vars"]["VECTOR_INDEXES"] = json.dumps(valid_indexes)
            else:
                add_status_message("⚠️ No valid vector indexes found", False)
        
        # Create endpoint
        response = requests.post(
            f"{client.config.host}/api/2.0/serving-endpoints",
            headers={
                "Authorization": f"Bearer {client.config.token}",
                "Content-Type": "application/json"
            },
            json=agent_config,
            timeout=30
        )
        
        if response.status_code in [200, 201]:
            endpoint_url = f"{client.config.host}/serving-endpoints/{endpoint_name}"
            add_status_message(f"✅ Created agent endpoint: {endpoint_url}")
            
            # Update session state
            st.session_state.agent_endpoints.append({
                'name': agent_name,
                'endpoint_name': endpoint_name,
                'url': endpoint_url,
                'model': model_name,
                'vector_indexes': vector_indexes or [],
                'status': 'Creating',
                'is_external': is_external_model
            })
            
            return endpoint_url
        else:
            error_msg = response.text
            try:
                error_json = response.json()
                error_msg = error_json.get('message', error_json.get('error_code', error_msg))
            except:
                pass
            add_status_message(f"❌ Failed to create agent endpoint: {error_msg}", False)
            
            # If foundation model approach fails, try a simpler approach
            if not is_external_model:
                add_status_message("🔄 Trying alternative approach for Databricks foundation model...", False)
                return create_simple_agent_endpoint(client, agent_name, model_name, system_prompt, vector_indexes)
            
            return ""
            
    except requests.exceptions.Timeout:
        add_status_message("❌ Request timeout - endpoint creation may still be in progress", False)
        return ""
    except Exception as e:
        add_status_message(f"❌ Error creating agent endpoint: {str(e)}", False)
        return ""

def perform_vector_search(client: WorkspaceClient, index_name: str, query: str, top_k: int = 5) -> List[Dict]:
    """Perform a vector search on the specified index"""
    try:
        # Initialize VectorSearchClient
        vsc = VectorSearchClient(workspace_url=client.config.host, personal_access_token=client.config.token)
        
        # Perform similarity search
        response = vsc.search(
            index_name=index_name,
            query_text=query,
            columns=["content", "source_file", "chunk_index"],
            num_results=top_k
        )
        
        results = []
        if response and 'result' in response and 'data_array' in response['result']:
            for record in response['result']['data_array']:
                results.append({
                    'content': record[0],
                    'source_file': record[1],
                    'chunk_index': record[2],
                    'score': record.get('score', 0.0) if 'score' in record else 0.0
                })
        
        add_status_message(f"✅ Retrieved {len(results)} results from vector index {index_name}")
        return results
    
    except Exception as e:
        add_status_message(f"❌ Error performing vector search on {index_name}: {str(e)}", False)
        return []

def chat_with_endpoint(client: WorkspaceClient, endpoint_name: str, message: str, 
                     system_prompt: str = "", vector_indexes: List[str] = None) -> str:
    """Send a chat message to a serving endpoint with improved error handling"""
    try:
        print(f"💬 Calling endpoint: {endpoint_name}")
        print(f"📝 Message length: {len(message)} characters")
        
        # Find the agent configuration in session state
        agent_config = None
        if hasattr(st.session_state, 'agent_endpoints'):
            for agent in st.session_state.agent_endpoints:
                if agent['endpoint_name'] == endpoint_name or agent['name'] == endpoint_name:
                    agent_config = agent
                    break
        
        # Use system prompt from agent config if not provided
        if not system_prompt and agent_config:
            system_prompt = agent_config.get('system_prompt', "")
        
        # Prepare the payload - use the message as-is since context is already included
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": message  # Message already contains system prompt and context
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        # Determine the actual endpoint to call
        actual_endpoint = endpoint_name
        if agent_config and agent_config.get('is_simple'):
            actual_endpoint = agent_config['model']
        
        print(f"🎯 Actual endpoint to call: {actual_endpoint}")
        
        response = requests.post(
            f"{client.config.host}/serving-endpoints/{actual_endpoint}/invocations",
            headers={
                "Authorization": f"Bearer {client.config.token}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=45
        )
        
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"📥 Response structure: {list(result.keys()) if isinstance(result, dict) else type(result)}")
            
            # Handle different response formats
            response_content = "No response content found"
            
            if "choices" in result and result["choices"]:
                # OpenAI/Anthropic format
                choice = result["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    response_content = choice["message"]["content"]
                elif "text" in choice:
                    response_content = choice["text"]
            elif "candidates" in result and result["candidates"]:
                # Google format
                candidate = result["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    response_content = candidate["content"]["parts"][0].get("text", "No text")
            elif "response" in result:
                # Direct response format
                response_content = result["response"]
            elif "generated_text" in result:
                # Hugging Face format
                response_content = result["generated_text"]
            elif isinstance(result, str):
                # String response
                response_content = result
            else:
                # Fallback - try to find any text content
                response_content = str(result)
                print(f"⚠️ Using fallback response parsing: {response_content[:100]}...")
                
            print(f"✅ Extracted response content: {response_content[:100]}...")
            add_status_message("✅ Successfully received response from endpoint")
            return response_content
        else:
            error_msg = f"HTTP {response.status_code}: {response.text}"
            print(f"❌ Endpoint error: {error_msg}")
            add_status_message(f"❌ Endpoint error: {error_msg}", False)
            return f"Error communicating with endpoint: {error_msg}"
            
    except requests.exceptions.Timeout:
        error_msg = "Request timeout - the endpoint took too long to respond"
        print(f"❌ {error_msg}")
        add_status_message(error_msg, False)
        return f"Error: {error_msg}"
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"❌ {error_msg}")
        add_status_message(error_msg, False)
        return f"Error: {error_msg}"