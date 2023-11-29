import os

from typing import Any, List, Optional
from typing import Any, Dict, List, Optional, Union
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult

os.environ["LANGCHAIN_TRACING"] = "true"

def _print_lines(text, line_prefix="\t\t| "):
    for line in text.split("\n"):
        print(line_prefix + line)


def _print_prompts(prompts):
    """
    Prints the prompts to the console.

    Args:
        prompts (List[str]): The prompts to print.

    Returns:
        None

    """
    if len(prompts) == 1:
        # If there is only one prompt, print it without formatting
        prompt = prompts[0]
        if len(prompt.split("\n")) == 1:
            print(f"\tprompt='{prompt}'")
        else:
            print("\tprompt=")
            _print_lines(prompts[0])
    else:
        # If there are multiple prompts, print them with formatting
        print(f"\tprompts={len(prompts)}")
        for i, prompt in enumerate(prompts):
            print(f"\n\tprompt[{i}]")
            _print_lines(prompt)

def _print_generations(generations):
    print(f"generations: {generations}")
    if len(generations) == 1:
        txt = generations[0].text
        if len(txt.split("\n")) == 1:
            print(f"\tgeneration='{txt}'")
        else:
            print("\tgeneration=")
            _print_lines(txt)
    else:
        print(f"\tgenerations={len(generations)}")
        for i, generation in enumerate(generations):
            print(f"\n\tgeneration[{i}]")
            _print_lines(generation.text)


class DebugCallbackHandler(BaseCallbackHandler):
    """
    Callback Handler that prints to std out
    """

    def __init__(self, color: Optional[str] = None) -> None:
        """Initialize callback handler."""
        self.color = color
        self.debug = True

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Print out the prompts."""
        if self.debug:
            print(f"on_llm_start:\n\tserialized={serialized}")
            _print_prompts(prompts)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Do nothing."""
        if self.debug:
            print(f"on_llm_end:\n\tllm_output: {response.llm_output}")
            for gens in response.generations:
                _print_generations(gens)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Do nothing."""
        if self.debug:
            print(f"on_llm_new_token: token={token}")

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Print out that we are entering a chain."""
        if self.debug:
            print(f"on_chain_start:\n\tserialized: {serialized}\n\tinputs: {inputs}")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain."""
        if self.debug:
            print(f"on_chain_end:\n\toutputs={outputs}")

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        pass

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Do nothing."""
        if self.debug:
            print(f"on_tool_start:\n\tserialized={serialized}\n\tinput_str={input_str}")

    def on_agent_action(
        self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Run on agent action."""
        if self.debug:
            print(f"on_agent_action:\n\taction={action}")

    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """If not the final action, print out observation."""
        if self.debug:
            print(f"on_tool_end:\n\toutput={output}\n\tobservation_prefix={observation_prefix}\n\tllm_prefix={llm_prefix}")

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        pass

    def on_text(
        self,
        text: str,
        color: Optional[str] = None,
        end: str = "",
        **kwargs: Any,
    ) -> None:
        """Run when agent ends."""
        if self.debug:
            print(f"on_text:\n\ttext={text}\n\tend={end}")

    def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Run on agent end."""
        if self.debug:
            print(f"on_agent_finish:\n\tfinish={finish}")