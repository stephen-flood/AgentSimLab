from __future__ import annotations
import yaml # used to format prompts

"""llm_models.py
Refactored, library‑agnostic wrapper classes for interacting with various LLM
back‑ends **while preserving the original public API**.  Concrete subclasses
handle provider‑specific details; the shared logic never inspects response
objects directly.

Public surface
--------------
* ``generate_content`` (alias ``generate``)
* ``register_tool`` / ``get_tool``
* ``apply_tool``
* ``print_response``
* Gemini‑specific ``multimodal_query`` retained

Internal abstraction points
---------------------------
* ``_generate``            – provider API call
* ``_create_tool_object``  – translate tool metadata
* ``_iter_tool_calls``     – yield ``(name, args, raw)`` triples
* ``_response_text``       – extract printable text
"""

from abc import ABC, abstractmethod
from datetime import datetime
import json
import os
import time
from typing import Any, Callable, Dict, List, Tuple, Sequence

__all__ = [
    "RateLimitTracker",
    "SimpleModel",
    "GeminiModel",
    "OpenApiModel",
]

###############################################################################
# Loads API key
###############################################################################
try:
  from google.colab import userdata
  os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")
except:
  with open("gemini.api_key") as file:
    os.environ["GEMINI_API_KEY"] = file.read()

###############################################################################
# Utility
###############################################################################

class RateLimitTracker:
    """Throttle helper for free‑tier endpoints."""

    def __init__(self, per_min_limit: int):
        self.per_min_limit = per_min_limit
        self._history: List[datetime] = []

    def log_query(self) -> None:
        now = datetime.now()
        self._history.append(now)
        self._history = [t for t in self._history if (now - t).seconds <= 60]

    def print(self):
        print(self._history)

    def time_to_wait(self) -> float:
        average_wait = 60 / self.per_min_limit
        n = len(self._history)
        # Spend the first quarter of the budget immediately
        # Spend the second quarter of the budget a bit quickly
        # Spend spend the second half slowly enough to avoid rate limiting
        if n < self.per_min_limit // 4:
            return 0.0
        if n < self.per_min_limit // 2:
            return average_wait / 2
        return average_wait * 2

###############################################################################
# Abstract base
###############################################################################

class SimpleModel(ABC):
    """Provider‑agnostic wrapper exposing the legacy API."""

    def __init__(
        self,
        model_name: str,
        per_min_limit: int ,
        per_day_limit: int ,
    ) -> None:
        self.model_name = model_name
        self._rate = RateLimitTracker(per_min_limit)
        self.per_day_limit = per_day_limit
        self._tool_registry: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def generate_content(self, **kwargs):
        self._rate.log_query()
        wait = self._rate.time_to_wait()
        if wait:
            time.sleep(wait)
        # Call subclass's version of llm query
        return self._generate(**kwargs)

    generate = generate_content  # ergonomic alias

    # .................................................................
    def register_tool(
        self,
        func: Callable[..., Any],
        description: str,
        parameters: Dict[str, Dict[str, Any]],
    ) -> Any:
        name = func.__name__
        if name in self._tool_registry:
            print(f"Warning: Tool {name} already registered. Using PREVIOUS definition.")
            return self._tool_registry[name]["tool"]
        # call subclass appropriate function to create the tool object itself
        tool_obj = self._create_tool_object(name, description, parameters)
        #
        self._tool_registry[name] = {
            "function": func, 
            "tool": tool_obj, 
            "description": description, 
            "parameters": parameters}
        return tool_obj

    def get_tool(self, name: str):
        if name not in self._tool_registry:
            print(f"Error: tool {name} is not registered")
        return self._tool_registry.get(name, {}).get("tool")

    # .................................................................
    def apply_tool(self, response, **kwargs):
        # Use subclass's function to return iterable of tool calls
        calls = self._iter_tool_calls(response)
        if not calls:
            return [("No Tool Calls", "")]
        results: List[Tuple[Any, Any]] = []
        for name, args, raw in calls:
            # Pass both the LLM supplied arguments AND the user defined **kwargs 
            merged = {**args, **kwargs}
            if name in self._tool_registry:
                out = self._tool_registry[name]["function"](**merged)
                results.append((out, raw))
            else:
                results.append((f"Tool {name} not registered", raw))
        return results

    def print_response(self, response):
        calls = self._iter_tool_calls(response)
        if calls:
            print([raw for *_unused, raw in calls])
        else:
            print(self._response_text(response))

    # ------------------------------------------------------------------
    # Abstract hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _generate(self, **kwargs):
        """
        LLM specific code to generate response
         - "user_prompt" for a text prompt
        """
        pass

    @abstractmethod
    def _create_tool_object(self, name: str, description: str, parameters: dict) -> Any:
        """
        LLM specific code to generate a tool object from the name, description, and parameters
        """
        pass

    @abstractmethod
    def _iter_tool_calls(self, response) -> List[Tuple[str, Dict[str, Any], Any]]:
        """
        LLM specific code to extract iterable of tool calls from LLM response
        """
        pass

    @abstractmethod
    def _response_text(self, response) -> str:
        """
        LLM specific code to extract response text from LLM response
        """
        pass

###############################################################################
# Google Gemini
###############################################################################

from google import genai
from google.genai import types as gtypes

class GeminiModel(SimpleModel):
    """Wrapper for *google‑genai*."""

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash-lite",
        per_min_limit: int = 30,
        per_day_limit: int = 1000,
        # **client_kw: Any,
    ) -> None:
        super().__init__(model_name, per_min_limit, per_day_limit)
        # if genai is None:
        #     raise ImportError("google-genai not installed – `pip install google-genai`.")
        # api_key = client_kw.pop("api_key", None) or os.getenv("GEMINI_API_KEY")
        # self._client = genai.Client(api_key=api_key, **client_kw)
        self.API_KEY = os.environ.get("GEMINI_API_KEY")
        self._client = genai.Client(
            api_key=self.API_KEY,
        )


    # Provider hooks ----------------------------------------------------
    ## NOTE: A lone * in a function definition indicates that
    #        all parameters after it must be passed as keyword 
    #        arguments
    # BUT: Here they have default values of None, so they are actually optional?
    def _generate(
        self,
        *,
        contents: List[ gtypes.Content] | None = None,
        user_prompt: str | None = None,
        tools: List[Any] | None = None,
        **_,
    ) -> gtypes.GenerateContentResponse:  # type: ignore
        if contents is None:
            if user_prompt is None:
                raise ValueError("GeminiModel.generate_content requires `user_prompt` or `contents`.")
            contents = [
                gtypes.Content(role="user", parts=[gtypes.Part.from_text(text = user_prompt)])
            ]
        cfg = gtypes.GenerateContentConfig(tools=tools, response_mime_type="text/plain")
        return self._client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=cfg,
        )

    def _create_tool_object(self, name: str, description: str, parameters: dict) -> gtypes.Tool:  # type: ignore
        return gtypes.Tool(
            function_declarations=[
                gtypes.FunctionDeclaration(
                    name=name,
                    description=description,
                    parameters=gtypes.Schema(
                        type=gtypes.Type.OBJECT,
                        properties=parameters,
                        required=list(parameters.keys()),
                    ),
                )
            ]
        )

    def _iter_tool_calls(self, response):
        calls = getattr(response, "function_calls", [])
        return [(c.name, c.args, c) for c in calls]

    def _response_text(self, response):
        return getattr(response, "text", "")

    # Legacy helper retained -------------------------------------------
    def multimodal_query(
        self,
        *,
        user_prompt: str,
        attachment_names: List[str] | None = None,
        system_prompt: str | None = None,
    ):
        if gtypes is None:
            raise ImportError("google‑genai types unavailable.")
        attachment_names = attachment_names or []
        parts: List[Any] = [self._part_from_ref(r) for r in attachment_names]
        if system_prompt:
            parts.append(gtypes.Content(role="system", parts=[gtypes.Part.from_text(text=system_prompt)]))
        parts.append(user_prompt)
        return self.generate_content(contents=parts)

    @staticmethod
    def _part_from_ref(ref: str):
        from urllib.parse import urlparse
        import mimetypes, requests
        is_url = urlparse(ref).scheme in ("http", "https")
        if is_url:
            if "youtube.com" in ref:
                return gtypes.Part.from_uri(file_uri=ref, mime_type="video/mp4")
            mime, _ = mimetypes.guess_type(ref)
            return gtypes.Part.from_bytes(data=requests.get(ref).content, mime_type=mime)
        mime, _ = mimetypes.guess_type(ref)
        with open(ref, "rb") as fh:
            return gtypes.Part.from_bytes(data=fh.read(), mime_type=mime)

###############################################################################
# OpenAI ChatCompletion
###############################################################################

import openai

class OpenApiModel(SimpleModel):
    """Wrapper for *openai* ChatCompletion / GPT‑4o."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        per_min_limit: int = 60,
        api_key: str | None = None,
        **client_kw: Any,
    ) -> None:
        super().__init__(model_name, per_min_limit)
        if openai is None:
            raise ImportError("openai not installed – `pip install openai`.")
        openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client_kw = client_kw

    # Provider hooks ----------------------------------------------------
    def _generate(
        self,
        *,
        messages: List[Dict[str, str]] | None = None,
        user_prompt: str | None = None,
        tools: List[Dict[str, Any]] | None = None,
        **kwargs,
    ):
        if messages is None:
            if user_prompt is None:
                raise ValueError("OpenApiModel.generate_content requires `user_prompt` or `messages`.")
            messages = [{"role": "user", "content": user_prompt}]
        return openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            **self._client_kw,
            **kwargs,
        )

    def _create_tool_object(self, name: str, description: str, parameters: dict) -> dict:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": list(parameters.keys()),
                },
            },
        }

    def _iter_tool_calls(self, response):
        tool_calls: List[Tuple[str, Dict[str, Any], Any]] = []
        for choice in getattr(response, "choices", []):
            msg = choice.message
            for tc in getattr(msg, "tool_calls", []):
                # tc.function.arguments is a JSON string in the new SDK
                args_str = tc.function.arguments
                try:
                    args_dict = json.loads(args_str) if isinstance(args_str, str) else args_str
                except Exception:
                    args_dict = {}
                tool_calls.append((tc.function.name, args_dict, tc))
        return tool_calls

    def _response_text(self, response):
        choices = getattr(response, "choices", [])
        if choices:
            return choices[0].message.content or ""
        return ""

###############################################################################
# Prompt Class (handwritten)
###############################################################################

class Prompt:
  """
  Assembles and formats prompts with containing the following parts
  persona, context, instruction, input, tone, output_format, examples
  """
  def __init__(self, **kwargs):

    self.details = {}
    self.verbose = False # for testing/debugging

    # Put standard prompt elements in a specific order
    prompt_elements = ["persona","context","instruction","input","tone","output_format","examples"]
    for element in prompt_elements:
      if element in kwargs:
        name = element.title()
        name = name.replace("_"," ")
        # self.details.append({name: kwargs[element]})
        self.details[name] = kwargs[element]

    # Add all other prompt elements to the end 
    for key,val in kwargs.items():
      if key not in prompt_elements:
        self.details[key] = val

  def generate(self):
    prompt = yaml.dump(self.details, sort_keys=False)
    # Repeat the instruction at the end for emphasis
    if "instruction" in self.details:
      prompt += f"\nRemember: the instruction is {self.details['instruction']}"

    if self.verbose:
        print("=======BEGIN_PROMPT=======")
        print(prompt)
        print("======= END PROMPT =======")
    return prompt
