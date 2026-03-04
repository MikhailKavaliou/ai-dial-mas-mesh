from typing import Any

from task.tools.deployment.base_agent_tool import BaseAgentTool


class CalculationsAgentTool(BaseAgentTool):

    @property
    def deployment_name(self) -> str:
        return "calculations-agent"

    @property
    def name(self) -> str:
        return "calculations_agent_tool"

    @property
    def description(self) -> str:
        return "Delegates tasks to the Calculations Agent for math, data analysis, and plotting."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Request for the Calculations Agent"
                },
                "propagate_history": {
                    "type": "boolean",
                    "description": "Whether to forward prior peer-to-peer conversation history",
                    "default": False
                }
            },
            "required": ["prompt"]
        }
