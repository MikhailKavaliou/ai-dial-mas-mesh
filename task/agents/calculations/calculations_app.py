import asyncio
import os

import uvicorn
from aidial_sdk import DIALApp
from aidial_sdk.chat_completion import ChatCompletion, Request, Response

from task.agents.calculations.calculations_agent import CalculationsAgent
from task.agents.calculations.tools.simple_calculator_tool import SimpleCalculatorTool
from task.tools.base_tool import BaseTool
from task.agents.calculations.tools.py_interpreter.python_code_interpreter_tool import PythonCodeInterpreterTool
from task.tools.deployment.content_management_agent_tool import ContentManagementAgentTool
from task.tools.deployment.web_search_agent_tool import WebSearchAgentTool
from task.tools.mcp.mcp_client import MCPClient
from task.utils.constants import DIAL_ENDPOINT, DEPLOYMENT_NAME

_PYTHON_MCP_URL = os.getenv('PYTHON_INTERPRETER_MCP_URL', "http://localhost:8050/mcp")
_PYTHON_TOOL_NAME = os.getenv('PYTHON_INTERPRETER_MCP_TOOL_NAME')


class CalculationsApplication(ChatCompletion):

    def __init__(self, agent: CalculationsAgent, mcp_client: MCPClient):
        self._agent = agent
        self._mcp_client = mcp_client

    @classmethod
    async def create(cls) -> "CalculationsApplication":
        mcp_client = await MCPClient.create(_PYTHON_MCP_URL)
        mcp_tools = await mcp_client.get_tools()
        if not mcp_tools:
            raise ValueError("Python interpreter MCP server returned no tools")

        selected_tool_name = _PYTHON_TOOL_NAME or mcp_tools[0].name
        python_tool = PythonCodeInterpreterTool(
            mcp_client=mcp_client,
            mcp_tool_models=mcp_tools,
            tool_name=selected_tool_name,
            dial_endpoint=DIAL_ENDPOINT,
        )

        tools: list[BaseTool] = [
            SimpleCalculatorTool(),
            python_tool,
            ContentManagementAgentTool(endpoint=DIAL_ENDPOINT),
            WebSearchAgentTool(endpoint=DIAL_ENDPOINT),
        ]
        return cls(
            agent=CalculationsAgent(endpoint=DIAL_ENDPOINT, tools=tools),
            mcp_client=mcp_client,
        )

    async def chat_completion(self, request: Request, response: Response) -> None:
        choice = response.create_single_choice()
        await self._agent.handle_request(
            deployment_name=DEPLOYMENT_NAME,
            choice=choice,
            request=request,
            response=response,
        )


async def create_app() -> DIALApp:
    return DIALApp(
        deployment_name="calculations-agent",
        impl=await CalculationsApplication.create(),
    )


if __name__ == "__main__":
    uvicorn.run(asyncio.run(create_app()), host="0.0.0.0", port=5001)
