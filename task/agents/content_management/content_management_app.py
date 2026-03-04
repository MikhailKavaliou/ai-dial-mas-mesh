import uvicorn
from aidial_sdk import DIALApp
from aidial_sdk.chat_completion import ChatCompletion, Request, Response

from task.agents.content_management.content_management_agent import ContentManagementAgent
from task.agents.content_management.tools.files.file_content_extraction_tool import FileContentExtractionTool
from task.agents.content_management.tools.rag.document_cache import DocumentCache
from task.agents.content_management.tools.rag.rag_tool import RagTool
from task.tools.base_tool import BaseTool
from task.tools.deployment.calculations_agent_tool import CalculationsAgentTool
from task.tools.deployment.web_search_agent_tool import WebSearchAgentTool
from task.utils.constants import DIAL_ENDPOINT, DEPLOYMENT_NAME


class ContentManagementApplication(ChatCompletion):

    def __init__(self, agent: ContentManagementAgent):
        self._agent = agent

    @classmethod
    async def create(cls) -> "ContentManagementApplication":
        document_cache = DocumentCache()
        tools: list[BaseTool] = [
            FileContentExtractionTool(endpoint=DIAL_ENDPOINT),
            RagTool(
                endpoint=DIAL_ENDPOINT,
                deployment_name=DEPLOYMENT_NAME,
                document_cache=document_cache,
            ),
            CalculationsAgentTool(endpoint=DIAL_ENDPOINT),
            WebSearchAgentTool(endpoint=DIAL_ENDPOINT),
        ]
        return cls(agent=ContentManagementAgent(endpoint=DIAL_ENDPOINT, tools=tools))

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
        deployment_name="content-management-agent",
        impl=await ContentManagementApplication.create(),
    )


if __name__ == "__main__":
    import asyncio
    uvicorn.run(asyncio.run(create_app()), host="0.0.0.0", port=5002)
