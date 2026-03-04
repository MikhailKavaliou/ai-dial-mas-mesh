import json
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Message, Role, CustomContent, Stage, Attachment
from pydantic import StrictStr

from task.tools.base_tool import BaseTool
from task.tools.models import ToolCallParams
from task.utils.constants import TOOL_CALL_HISTORY_KEY
from task.utils.stage import StageProcessor


class BaseAgentTool(BaseTool, ABC):

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    @property
    @abstractmethod
    def deployment_name(self) -> str:
        pass

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        client = AsyncDial(
            base_url=self.endpoint,
            api_key=tool_call_params.api_key,
            api_version='2025-01-01-preview'
        )

        chunks = await client.chat.completions.create(
            messages=self._prepare_messages(tool_call_params),
            stream=True,
            deployment_name=self.deployment_name,
            extra_headers={"x-conversation-id": tool_call_params.conversation_id}
        )

        content = ''
        custom_content: CustomContent = CustomContent(attachments=[])
        stages_map: dict[int, Stage] = {}
        known_attachment_keys: set[tuple[str, str, str]] = set()

        try:
            async for chunk in chunks:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                if not delta:
                    continue

                if delta.content:
                    content += delta.content
                    if tool_call_params.stage:
                        tool_call_params.stage.append_content(delta.content)

                delta_custom_content = self._to_dict(getattr(delta, "custom_content", None))
                if not delta_custom_content:
                    continue

                state = delta_custom_content.get("state")
                if isinstance(state, dict):
                    custom_content.state = state

                for raw_attachment in delta_custom_content.get("attachments") or []:
                    attachment = self._to_attachment(raw_attachment)
                    if not attachment:
                        continue

                    key = self._attachment_key(attachment)
                    if key in known_attachment_keys:
                        continue

                    known_attachment_keys.add(key)
                    custom_content.attachments.append(attachment)
                    tool_call_params.choice.add_attachment(attachment)

                for raw_stage in delta_custom_content.get("stages") or []:
                    if not isinstance(raw_stage, dict):
                        continue

                    stage_index = raw_stage.get("index")
                    if not isinstance(stage_index, int):
                        continue

                    propagated_stage = stages_map.get(stage_index)
                    if not propagated_stage:
                        propagated_stage = StageProcessor.open_stage(
                            choice=tool_call_params.choice,
                            name=raw_stage.get("name")
                        )
                        stages_map[stage_index] = propagated_stage

                    if raw_stage.get("content"):
                        propagated_stage.append_content(raw_stage["content"])

                    for raw_attachment in raw_stage.get("attachments") or []:
                        attachment = self._to_attachment(raw_attachment)
                        if attachment:
                            propagated_stage.add_attachment(attachment)

                    if raw_stage.get("status") == "completed":
                        StageProcessor.close_stage_safely(propagated_stage)
        finally:
            for propagated_stage in stages_map.values():
                StageProcessor.close_stage_safely(propagated_stage)

        return Message(
            role=Role.TOOL,
            name=StrictStr(tool_call_params.tool_call.function.name),
            tool_call_id=StrictStr(tool_call_params.tool_call.id),
            content=StrictStr(content),
            custom_content=custom_content
        )

    def _prepare_messages(self, tool_call_params: ToolCallParams) -> list[dict[str, Any]]:
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        prompt = arguments["prompt"]
        propagate_history = arguments.get("propagate_history", False)

        messages: list[dict[str, Any]] = []
        request_messages = tool_call_params.messages

        if propagate_history:
            for index, message in enumerate(request_messages):
                if message.role != Role.ASSISTANT or not message.custom_content:
                    continue

                state = message.custom_content.state
                if not isinstance(state, dict):
                    continue

                agent_state = state.get(self.name)
                if not isinstance(agent_state, dict):
                    continue

                history = agent_state.get(TOOL_CALL_HISTORY_KEY)
                if not isinstance(history, list) or not history:
                    continue

                if index > 0 and request_messages[index - 1].role == Role.USER:
                    messages.append(request_messages[index - 1].dict(exclude_none=True))

                assistant_message = deepcopy(message)
                assistant_message.custom_content.state = agent_state
                messages.append(assistant_message.dict(exclude_none=True))

        final_user_message: dict[str, Any] = {
            "role": Role.USER.value,
            "content": prompt,
        }

        attachments = self._latest_user_attachments(request_messages)
        if attachments:
            final_user_message["custom_content"] = {
                "attachments": [attachment.dict(exclude_none=True) for attachment in attachments]
            }

        messages.append(final_user_message)
        return messages

    @staticmethod
    def _latest_user_attachments(messages: list[Message]) -> list[Attachment]:
        for message in reversed(messages):
            if message.role == Role.USER and message.custom_content and message.custom_content.attachments:
                return message.custom_content.attachments
        return []

    @staticmethod
    def _to_dict(value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        if isinstance(value, dict):
            return value
        if hasattr(value, "model_dump"):
            return value.model_dump(exclude_none=True)
        if hasattr(value, "dict"):
            return value.dict(exclude_none=True)
        return None

    @staticmethod
    def _to_attachment(raw_attachment: Any) -> Attachment | None:
        if isinstance(raw_attachment, Attachment):
            return raw_attachment
        if isinstance(raw_attachment, dict):
            if hasattr(Attachment, "model_validate"):
                return Attachment.model_validate(raw_attachment)
            return Attachment.validate(raw_attachment)
        return None

    @staticmethod
    def _attachment_key(attachment: Attachment) -> tuple[str, str, str]:
        return (
            attachment.url or '',
            attachment.reference_url or '',
            attachment.title or ''
        )
