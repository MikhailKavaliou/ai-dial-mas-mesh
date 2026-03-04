from typing import Any

from task.tools.deployment.base_agent_tool import BaseAgentTool


class WebSearchAgentTool(BaseAgentTool):

    @property
    def deployment_name(self) -> str:
        return "web-search-agent"

    @property
    def name(self) -> str:
        return "web_search_agent_tool"

    @property
    def description(self) -> str:
        return "Delegates tasks to the Web Search Agent for online research and page retrieval."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Request for the Web Search Agent"
                },
                "propagate_history": {
                    "type": "boolean",
                    "description": "Whether to forward prior peer-to-peer conversation history",
                    "default": False
                }
            },
            "required": ["prompt"]
        }
