from typing import Any, Dict, List, Optional

SYSTEM_PROMPT = (
    "You are a helpful assistant, one of the greatest AI scientists, logicians and mathematicians. "
    "Make sure you carefully and fully understand the details of user's requirements before you "
    "start solving the problem."
)


class ProviderModel:
    """Adapter that lets Aristotle query the language model through an ``LLMProvider``.

    Parameters
    ----------
    llm : LLMProvider
        Any object with a ``model_name`` attribute and an ``invoke(messages)`` method
        that returns an object with a ``response`` string (``LLMOut``).
    system_prompt : str, default=SYSTEM_PROMPT
        System message sent before every prompt (the one used by the original code).
    stop : str or None, default="------"
        The prompts separate their examples with this string. The response is cut
        there, as the original API call did with its stop sequence.

    Notes
    -----
    Anything up to a closing ``</think>`` tag is dropped, so the parsers only see the
    final answer of reasoning models.
    """

    def __init__(
        self,
        llm: Any,
        system_prompt: str = SYSTEM_PROMPT,
        stop: Optional[str] = "------",
    ):
        self.llm = llm
        self.name = llm.model_name
        self.system_prompt = system_prompt
        self.stop = stop

    def messages(self, prompt: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

    def generate(self, prompt: str) -> str:
        return self.clean(self.llm.invoke(self.messages(prompt)).response)

    def clean(self, text: str) -> str:
        text = text.split("</think>")[-1]
        if self.stop:
            text = text.split(self.stop)[0]
        return text.strip()
