python -m venv .venv
macOS/Linux: source .venv/bin/activate
Windows CMD: .venv\Scripts\activate.bat
Windows PowerShell: .venv\Scripts\Activate.ps1

pip install google-adk
mkdir multi_tool_agent/
echo "from . import agent" > multi_tool_agent/__init__.py
touch multi_tool_agent/agent.py
touch multi_tool_agent/.env

adk web
adk run multi_tool_agent
adk api_server

curl -X POST http://0.0.0.0:8000/apps/multi_tool_agent/users/u_123/sessions/s_123 \
  -H "Content-Type: application/json" \
  -d '{"state": {"key1": "value1", "key2": 42}}'

curl -X POST http://0.0.0.0:8000/run \
-H "Content-Type: application/json" \
-d '{
"app_name": "multi_tool_agent",
"user_id": "u_123",
"session_id": "s_123",
"new_message": {
    "role": "user",
    "parts": [{
    "text": "Hey whats the weather in new york today"
    }]
}
}'