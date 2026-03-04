from typing import Any

from task.tools.deployment.base_agent_tool import BaseAgentTool


class ContentManagementAgentTool(BaseAgentTool):

    @property
    def deployment_name(self) -> str:
        return "content-management-agent"

    @property
    def name(self) -> str:
        return "content_management_agent_tool"

    @property
    def description(self) -> str:
        return "Delegates tasks to the Content Management Agent for file extraction and document RAG."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Request for the Content Management Agent"
                },
                "propagate_history": {
                    "type": "boolean",
                    "description": "Whether to forward prior peer-to-peer conversation history",
                    "default": False
                }
            },
            "required": ["prompt"]
        }
