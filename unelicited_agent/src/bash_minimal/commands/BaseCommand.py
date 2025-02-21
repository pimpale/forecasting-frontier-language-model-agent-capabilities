import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

from inspect_ai.tool import Tool


@dataclass
class CommandOutput:
    content: str


CallbackType = Callable[[str, str], None]


class BaseCommand(ABC):
    def __init__(
        self, xml_tag: str, description: str, callback: Optional[CallbackType] = None
    ):
        self.xml_tag = xml_tag
        self.description = description
        self.callback = callback

    def count_occurrences(self, response: str) -> int:
        return len(self.extract_content(response))

    def extract_content(self, response: str) -> list[str]:
        """Extracts content from innermost XML-like tags in the response string."""
        pattern = re.compile(
            f"<{self.xml_tag}>((?:(?!<{self.xml_tag}>).)*?)</{self.xml_tag}>", re.DOTALL
        )
        return pattern.findall(response)

    def extract_full_tags(self, response: str) -> list[str]:
        contents = self.extract_content(response)
        return [f"<{self.xml_tag}>{content}</{self.xml_tag}>" for content in contents]

    @abstractmethod
    async def _run(self, content: str) -> CommandOutput:
        pass

    @abstractmethod
    def native_tool(self) -> Tool:
        pass

    async def execute(self, response: str) -> list[CommandOutput]:
        results = []
        contents = self.extract_content(response)
        for content in contents:
            result = await self._run(content)
            if self.callback:
                self.callback(content, result.content)
            results.append(result)
        return results

    def __str__(self):
        return f"{self.xml_tag}: {self.description}"
