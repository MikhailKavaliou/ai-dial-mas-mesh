import asyncio
import os

import uvicorn
from aidial_sdk import DIALApp
from aidial_sdk.chat_completion import ChatCompletion, Request, Response

from task.agents.web_search.web_search_agent import WebSearchAgent
from task.tools.base_tool import BaseTool
from task.tools.deployment.calculations_agent_tool import CalculationsAgentTool
from task.tools.deployment.content_management_agent_tool import ContentManagementAgentTool
from task.tools.mcp.mcp_client import MCPClient
from task.tools.mcp.mcp_tool import MCPTool
from task.utils.constants import DIAL_ENDPOINT, DEPLOYMENT_NAME

_DDG_MCP_URL = os.getenv('DDG_MCP_URL', "http://localhost:8051/mcp")


class WebSearchApplication(ChatCompletion):

    def __init__(self, agent: WebSearchAgent, mcp_client: MCPClient):
        self._agent = agent
        self._mcp_client = mcp_client

    @classmethod
    async def create(cls) -> "WebSearchApplication":
        mcp_client = await MCPClient.create(_DDG_MCP_URL)
        mcp_tools = await mcp_client.get_tools()

        tools: list[BaseTool] = [
            MCPTool(client=mcp_client, mcp_tool_model=mcp_tool_model)
            for mcp_tool_model in mcp_tools
        ]
        tools.extend([
            CalculationsAgentTool(endpoint=DIAL_ENDPOINT),
            ContentManagementAgentTool(endpoint=DIAL_ENDPOINT),
        ])

        return cls(
            agent=WebSearchAgent(endpoint=DIAL_ENDPOINT, tools=tools),
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
        deployment_name="web-search-agent",
        impl=await WebSearchApplication.create(),
    )


if __name__ == "__main__":
    uvicorn.run(asyncio.run(create_app()), host="0.0.0.0", port=5003)
