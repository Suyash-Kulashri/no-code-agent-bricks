import streamlit as st
import requests
import json
import time
from typing import Dict, List, Optional
from databricks.sdk import WorkspaceClient

def clear_current_step_messages():
    """Clear messages from current step"""
    if not hasattr(st.session_state, 'current_step_messages'):
        st.session_state.current_step_messages = []
    st.session_state.current_step_messages = []

def add_status_message(message: str, is_success: bool = True):
    """Add a status message to current step"""
    if not hasattr(st.session_state, 'current_step_messages'):
        st.session_state.current_step_messages = []
    
    st.session_state.current_step_messages.append({
        'message': message,
        'is_success': is_success,
        'timestamp': time.time()
    })
    
    # Display the message immediately
    if is_success:
        st.success(message)
    else:
        st.error(message)

def get_catalogs(client: WorkspaceClient) -> List[str]:
    """Get list of all catalogs"""
    try:
        catalogs = list(client.catalogs.list())
        return [catalog.name for catalog in catalogs if catalog.name]
    except Exception as e:
        add_status_message(f"❌ Error fetching catalogs: {str(e)}", False)
        return []

def get_schemas(client: WorkspaceClient, catalog_name: str) -> List[str]:
    """Get list of schemas in a catalog"""
    try:
        schemas = list(client.schemas.list(catalog_name=catalog_name))
        return [schema.name for schema in schemas if schema.name]
    except Exception as e:
        add_status_message(f"❌ Error fetching schemas: {str(e)}", False)
        return []

def get_foundation_models(client: WorkspaceClient) -> List[str]:
    """Get list of available Databricks foundation models for chat"""
    try:
        add_status_message("🔍 Fetching available foundation models...")
        
        # Get registered models first
        try:
            response = requests.get(
                f"{client.config.host}/api/2.0/mlflow/registered-models/list",
                headers={"Authorization": f"Bearer {client.config.token}"},
                timeout=15
            )
            
            registered_models = set()
            if response.status_code == 200:
                data = response.json()
                for model in data.get("registered_models", []):
                    name = model.get("name", "")
                    # Filter for chat/instruct models (exclude embedding models)
                    if any(keyword in name.lower() for keyword in ["instruct", "chat", "llama", "mixtral", "dbrx"]) and \
                       not any(exclude in name.lower() for exclude in ["embed", "bge", "e5"]):
                        registered_models.add(name)
            
            if registered_models:
                models_list = sorted(list(registered_models))
                add_status_message(f"✅ Found {len(models_list)} chat models")
                return models_list
                
        except Exception as e:
            add_status_message(f"⚠️ Could not fetch registered models: {str(e)}")
        
        # Fallback to common Databricks foundation models for chat
        add_status_message("📋 Using standard Databricks foundation model list")
        return [
            "databricks-dbrx-instruct",
            "databricks-meta-llama-3-1-405b-instruct",
            "databricks-meta-llama-3-1-70b-instruct",
            "databricks-meta-llama-3-70b-instruct",
            "databricks-mixtral-8x7b-instruct",
            "databricks-mpt-30b-instruct",
            "databricks-llama-2-70b-chat"
        ]
        
    except Exception as e:
        add_status_message(f"❌ Error fetching foundation models: {str(e)}, using defaults", False)
        return [
            "databricks-dbrx-instruct",
            "databricks-meta-llama-3-1-70b-instruct",
            "databricks-mixtral-8x7b-instruct"
        ]

def create_agent_endpoint(client: WorkspaceClient, agent_name: str, model_name: str, 
                        system_prompt: str, vector_indexes: List[str] = None, api_key: str = "") -> str:
    """Create an agent endpoint using the correct approach"""
    
    # Determine if this is an external model
    is_external = bool(api_key)
    
    if is_external:
        return create_external_model_endpoint(client, agent_name, model_name, system_prompt, api_key)
    else:
        return create_databricks_model_endpoint(client, agent_name, model_name, system_prompt, vector_indexes)

def create_databricks_model_endpoint(client: WorkspaceClient, agent_name: str, 
                                   model_name: str, system_prompt: str, 
                                   vector_indexes: List[str] = None) -> str:
    """Create a serving endpoint for Databricks foundation models"""
    try:
        endpoint_name = f"{agent_name.replace(' ', '_').lower()}_endpoint"
        add_status_message(f"🚀 Creating Databricks serving endpoint: {endpoint_name}")
        
        # Check if endpoint already exists
        if _endpoint_exists(client, endpoint_name):
            add_status_message(f"⚠️ Endpoint {endpoint_name} already exists", False)
            return ""
        
        # Create endpoint configuration for Databricks foundation model
        endpoint_config = {
            "name": endpoint_name,
            "config": {
                "served_entities": [{
                    "entity_name": model_name,
                    "entity_version": "1",
                    "workload_size": "Small",
                    "scale_to_zero_enabled": True,
                    "environment_vars": {
                        "SYSTEM_PROMPT": system_prompt[:1000],  # Limit length
                        "VECTOR_INDEXES": json.dumps(vector_indexes or [])
                    }
                }],
                "traffic_config": {
                    "routes": [{
                        "served_model_name": model_name.split("/")[-1],  # Use model name without prefix
                        "traffic_percentage": 100
                    }]
                }
            }
        }
        
        return _create_endpoint(client, endpoint_config, agent_name, model_name, system_prompt, vector_indexes, False)
        
    except Exception as e:
        add_status_message(f"❌ Error creating Databricks endpoint: {str(e)}", False)
        return ""

def create_external_model_endpoint(client: WorkspaceClient, agent_name: str, 
                                 model_name: str, system_prompt: str, api_key: str) -> str:
    """Create a serving endpoint for external models"""
    try:
        endpoint_name = f"{agent_name.replace(' ', '_').lower()}_endpoint"
        add_status_message(f"🌐 Creating external model endpoint: {endpoint_name}")
        
        # Check if endpoint already exists
        if _endpoint_exists(client, endpoint_name):
            add_status_message(f"⚠️ Endpoint {endpoint_name} already exists", False)
            return ""
        
        provider = _get_model_provider(model_name)
        
        # Create external model configuration
        external_model_config = {
            "name": model_name,
            "provider": provider,
            "task": "llm/v1/chat"
        }
        
        # Add provider-specific config
        if provider == "openai":
            external_model_config["openai_config"] = {"openai_api_key": api_key}
        elif provider == "anthropic":
            external_model_config["anthropic_config"] = {"anthropic_api_key": api_key}
        elif provider == "google":
            external_model_config["google_config"] = {"google_api_key": api_key}
        elif provider == "cohere":
            external_model_config["cohere_config"] = {"cohere_api_key": api_key}
        
        endpoint_config = {
            "name": endpoint_name,
            "config": {
                "served_entities": [{
                    "name": f"{agent_name}_entity",
                    "external_model": external_model_config
                }],
                "traffic_config": {
                    "routes": [{
                        "served_model_name": f"{agent_name}_entity",
                        "traffic_percentage": 100
                    }]
                }
            }
        }
        
        return _create_endpoint(client, endpoint_config, agent_name, model_name, system_prompt, [], True)
        
    except Exception as e:
        add_status_message(f"❌ Error creating external model endpoint: {str(e)}", False)
        return ""

def _endpoint_exists(client: WorkspaceClient, endpoint_name: str) -> bool:
    """Check if endpoint already exists"""
    try:
        response = requests.get(
            f"{client.config.host}/api/2.0/serving-endpoints/{endpoint_name}",
            headers={"Authorization": f"Bearer {client.config.token}"},
            timeout=10
        )
        return response.status_code == 200
    except:
        return False

def _create_endpoint(client: WorkspaceClient, endpoint_config: dict, agent_name: str, 
                    model_name: str, system_prompt: str, vector_indexes: List[str], 
                    is_external: bool) -> str:
    """Create the actual endpoint"""
    try:
        response = requests.post(
            f"{client.config.host}/api/2.0/serving-endpoints",
            headers={
                "Authorization": f"Bearer {client.config.token}",
                "Content-Type": "application/json"
            },
            json=endpoint_config,
            timeout=60  # Increased timeout
        )
        
        if response.status_code in [200, 201]:
            endpoint_name = endpoint_config["name"]
            endpoint_url = f"{client.config.host}/serving-endpoints/{endpoint_name}"
            
            add_status_message(f"✅ Serving endpoint created successfully!")
            add_status_message(f"🔗 Endpoint URL: {endpoint_url}")
            add_status_message("⏳ Endpoint is being deployed, this may take 5-10 minutes...")
            
            # Store endpoint info
            if not hasattr(st.session_state, 'custom_endpoints'):
                st.session_state.custom_endpoints = []
                
            st.session_state.custom_endpoints.append({
                'name': agent_name,
                'endpoint_name': endpoint_name,
                'url': endpoint_url,
                'model': model_name,
                'system_prompt': system_prompt,
                'vector_indexes': vector_indexes or [],
                'is_external': is_external,
                'status': 'Creating',
                'created_at': time.time()
            })
            
            return endpoint_url
            
        else:
            error_info = _parse_error_response(response)
            add_status_message(f"❌ Failed to create endpoint: {error_info}", False)
            
            # Provide specific guidance based on error
            if "RegisteredModel" in error_info and "does not exist" in error_info:
                add_status_message("💡 The selected model may not be available in your workspace. Try a different model.", False)
            elif "permission" in error_info.lower() or "access" in error_info.lower():
                add_status_message("💡 You may not have permission to create serving endpoints. Contact your workspace admin.", False)
            elif "quota" in error_info.lower() or "limit" in error_info.lower():
                add_status_message("💡 You may have reached serving endpoint limits. Delete unused endpoints first.", False)
            
            return ""
            
    except Exception as e:
        add_status_message(f"❌ Error creating endpoint: {str(e)}", False)
        return ""

def _get_model_provider(model_name: str) -> str:
    """Determine the model provider based on model name"""
    model_name_lower = model_name.lower()
    if "gpt" in model_name_lower or "openai" in model_name_lower:
        return "openai"
    elif "claude" in model_name_lower:
        return "anthropic"
    elif "gemini" in model_name_lower:
        return "google"
    elif "command" in model_name_lower:
        return "cohere"
    else:
        return "openai"  # default

def _parse_error_response(response) -> str:
    """Parse error response from API"""
    try:
        error_json = response.json()
        if "message" in error_json:
            return error_json["message"]
        elif "error_code" in error_json:
            return error_json["error_code"]
        elif "detail" in error_json:
            return error_json["detail"]
        else:
            return str(error_json)
    except:
        return f"HTTP {response.status_code}: {response.text[:200]}"

def check_endpoint_status(client: WorkspaceClient, endpoint_name: str) -> Dict:
    """Check the status of a serving endpoint"""
    try:
        response = requests.get(
            f"{client.config.host}/api/2.0/serving-endpoints/{endpoint_name}",
            headers={"Authorization": f"Bearer {client.config.token}"},
            timeout=15
        )
        
        if response.status_code == 200:
            endpoint_info = response.json()
            state = endpoint_info.get('state', {})
            
            # Get more detailed status
            ready_state = state.get('ready', 'Unknown')
            config_update = state.get('config_update', 'Unknown')
            
            # Determine overall status
            if ready_state == 'READY' and config_update == 'NOT_UPDATING':
                overall_status = 'Ready'
            elif config_update in ['UPDATING', 'PENDING']:
                overall_status = 'Updating'
            elif ready_state in ['NOT_READY', 'PROVISIONING']:
                overall_status = 'Creating'
            else:
                overall_status = f"{ready_state}"
            
            return {
                'status': overall_status,
                'ready_state': ready_state,
                'config_update': config_update,
                'url': f"{client.config.host}/serving-endpoints/{endpoint_name}",
                'details': state
            }
        else:
            return {'status': 'Error', 'error': _parse_error_response(response)}
            
    except Exception as e:
        return {'status': 'Error', 'error': str(e)}

def test_endpoint_chat(client: WorkspaceClient, endpoint_name: str, message: str, 
                      system_prompt: str = "") -> str:
    """Test chat with a serving endpoint"""
    try:
        add_status_message(f"💬 Testing endpoint: {endpoint_name}")
        
        # Check endpoint status first
        status = check_endpoint_status(client, endpoint_name)
        if status.get('status') != 'Ready':
            return f"Endpoint not ready. Status: {status.get('status', 'Unknown')}"
        
        # Prepare chat messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})
        
        payload = {
            "messages": messages,
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{client.config.host}/serving-endpoints/{endpoint_name}/invocations",
            headers={
                "Authorization": f"Bearer {client.config.token}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Parse response based on format
            content = ""
            if "choices" in result and result["choices"]:
                choice = result["choices"][0]
                if "message" in choice:
                    content = choice["message"].get("content", "")
                elif "text" in choice:
                    content = choice["text"]
            elif "candidates" in result and result["candidates"]:
                content = result["candidates"][0]["content"]["parts"][0]["text"]
            elif "response" in result:
                content = result["response"]
            elif "content" in result:
                content = result["content"]
            else:
                content = str(result)
                
            add_status_message("✅ Successfully received response from endpoint")
            return content.strip()
        else:
            error_msg = _parse_error_response(response)
            add_status_message(f"❌ Endpoint error: {error_msg}", False)
            return f"Error: {error_msg}"
            
    except Exception as e:
        add_status_message(f"❌ Error testing endpoint: {str(e)}", False)
        return f"Error: {str(e)}"

def list_vector_search_indexes(client: WorkspaceClient, catalog_name: str, schema_name: str) -> List[Dict]:
    """List vector search indexes in a catalog and schema"""
    try:
        # Get tables that might be vector indexes
        tables = list(client.tables.list(catalog_name=catalog_name, schema_name=schema_name))
        vector_indexes = []
        
        for table in tables:
            # Check if table has vector search capabilities
            table_info = {
                'name': table.name,
                'full_name': table.full_name,
                'catalog': catalog_name,
                'schema': schema_name,
                'comment': table.comment or 'No description',
                'table_type': table.table_type.value if table.table_type else 'Table'
            }
            
            # Only include tables that look like vector indexes
            if any(keyword in table.name.lower() for keyword in ['vector', 'index', 'embedding']) or \
               any(keyword in (table.comment or '').lower() for keyword in ['vector', 'embedding']):
                vector_indexes.append(table_info)
        
        return vector_indexes
        
    except Exception as e:
        add_status_message(f"❌ Error listing vector indexes: {str(e)}", False)
        return []

def get_available_serving_endpoints(client: WorkspaceClient) -> List[Dict]:
    """Get list of existing serving endpoints"""
    try:
        response = requests.get(
            f"{client.config.host}/api/2.0/serving-endpoints",
            headers={"Authorization": f"Bearer {client.config.token}"},
            timeout=15
        )
        
        endpoints = []
        if response.status_code == 200:
            data = response.json()
            for endpoint in data.get("endpoints", []):
                endpoints.append({
                    'name': endpoint.get('name', ''),
                    'state': endpoint.get('state', {}).get('ready', 'Unknown'),
                    'config_update': endpoint.get('state', {}).get('config_update', 'Unknown'),
                    'url': f"{client.config.host}/serving-endpoints/{endpoint.get('name', '')}"
                })
        
        return endpoints
        
    except Exception as e:
        add_status_message(f"❌ Error fetching serving endpoints: {str(e)}", False)
        return []

def delete_endpoint(client: WorkspaceClient, endpoint_name: str) -> bool:
    """Delete a serving endpoint"""
    try:
        response = requests.delete(
            f"{client.config.host}/api/2.0/serving-endpoints/{endpoint_name}",
            headers={"Authorization": f"Bearer {client.config.token}"},
            timeout=30
        )
        
        if response.status_code in [200, 204]:
            add_status_message(f"✅ Endpoint {endpoint_name} deleted successfully")
            return True
        else:
            error_msg = _parse_error_response(response)
            add_status_message(f"❌ Failed to delete endpoint: {error_msg}", False)
            return False
            
    except Exception as e:
        add_status_message(f"❌ Error deleting endpoint: {str(e)}", False)
        return False