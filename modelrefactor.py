from __future__ import annotations
import yaml # used to format prompts
import pprint

"""llm_models.py
Refactored, library-agnostic wrapper classes for interacting with various LLM
back-ends **while preserving the original public API**.  Concrete subclasses
handle provider-specific details; the shared logic never inspects response
objects directly.

Public surface
--------------
* ``generate_content`` (alias ``generate``)
* ``register_tool`` / ``get_tool``
* ``apply_tool``
* ``print_response``

Internal abstraction points
---------------------------
* ``_generate``            - provider API call
* ``_create_tool_object``  - translate tool metadata
* ``_iter_tool_calls``     - yield ``(name, args, raw)`` triples
* ``_response_text``       - extract printable text
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
    # "OpenApiModel",
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
    """Throttle helper for free-tier endpoints."""

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
        if self.per_min_limit < 0:
            return 0
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
    """Provider-agnostic wrapper exposing the legacy API."""

    def __init__(
        self,
        model_name: str,
        per_min_limit: int = -1,
        per_day_limit: int = -1,
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
        try:
            content = self._generate(**kwargs)
        except Exception as e:
            print("Error generating content:" , e)
            pprint.pp(kwargs)
            content = None
        return content

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
                try:    
                    out = self._tool_registry[name]["function"](**merged)
                    results.append((out, raw))
                except:
                    # raise ValueError(f"Error applying tool {name}", raw)
                    print(f"=========\n ERROR applying tool {name}.\n=========")
                    results.append((f"Error applying tool {name}. DOUBLE CHECK that your variables match the tool schema.", raw))
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
    """Wrapper for *google-genai*."""

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash-lite",
        per_min_limit: int = 30,
        per_day_limit: int = 1000,
        # **client_kw: Any,
    ) -> None:
        super().__init__(model_name, per_min_limit, per_day_limit)
        # if genai is None:
        #     raise ImportError("google-genai not installed - `pip install google-genai`.")
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
        # ── OLD parameters ───────────────────────────────────────────
        user_prompt: str | None = None,
        contents: List[gtypes.Content] | None = None,
        tools:    List[Any]           | None = None,
        # ── NEW multimodal hooks ─────────────────────────────────────
        attachment_names: List[str]   | None = None,
        system_prompt:    str         | None = None,
        **_,
    ) -> gtypes.GenerateContentResponse:          # type: ignore
        """
        Unified text + multimodal generation helper.

        You can call `generate_content` in *either* of two styles:

        • Plain-text:
            model.generate_content(user_prompt="Hello world")

        • Multimodal:
            model.generate_content(
                user_prompt      ="Describe the picture",
                attachment_names =["/path/dog.png", "https://…/cat.jpg"],
                system_prompt    ="You are a helpful image analyst"
            )
        """
        # ------------------------------------------------------------------
        # If the caller handed us a ready-made `contents` list, respect it.
        # Otherwise build one from the convenience kwargs.
        # ------------------------------------------------------------------
        if contents is None:
            contents_list: List[gtypes.Content | gtypes.Part] = []
            attachment_names = attachment_names or []

            # 1. binary/file/media parts
            for ref in attachment_names:
                contents_list.append(self._part_from_ref(ref))

            # 2. optional system instruction
            if system_prompt:
                contents_list.append(
                    gtypes.Content(
                        role="system",
                        parts=[gtypes.Part.from_text(text=system_prompt)],
                    )
                )

            # 3. primary user prompt (required in this code-path)
            if user_prompt is None:
                raise ValueError(
                    "GeminiModel.generate_content needs `user_prompt` "
                    "when `contents` is omitted."
                )
            contents_list.append(
                gtypes.Content(
                    role="user",
                    parts=[gtypes.Part.from_text(text=user_prompt)],
                )
            )
            contents = contents_list   # now fully assembled

        # ------------------------------------------------------------------
        cfg = gtypes.GenerateContentConfig(
            tools=tools,
            response_mime_type="text/plain",
        )
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
# HTTP based access 
###############################################################################
import json, requests
from typing import Any, Dict, List, Iterator, Tuple, Union

Message = Dict[str, Union[str, list, dict]]            # helper alias

class HTTPChatModel(SimpleModel):
    """
    Generic client for any OpenAI-compatible HTTP server (vLLM, Ollama, LocalAI …).
    Always sends tool specs under the **tools** key, wrapped with {"type": "function"}.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str                    = "http://localhost:11434/v1",
        api_key:   str | None            = None,
        **opts: Any,
    ) -> None:
        super().__init__(model_name, **opts)
        self.base_url    = base_url.rstrip("/")
        self.api_key     = api_key or ""
        self._functions: List[Dict[str, Any]] = []      # raw function specs

        self.session     = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    # ------------------------------------------------------------------
    # Same signature as SimpleModel / GeminiModel
    # ------------------------------------------------------------------
    def _create_tool_object(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build an OpenAI-style function spec.  Stored in self._functions and later
        wrapped into the `tools` field at call-time.
        """
        # spec = {
        #     "name": name, 
        #     "description": description, 
        #     "parameters": parameters,
        #     "required" : list(parameters.keys())
        # }
        spec: Dict[str, Any] = {
            "name":        name,
            "description": description,
            "parameters": {
                "type":       "object",
                "properties": parameters,
                "required" : list(parameters.keys()),
            },
        }
        self._functions.append(spec)
        return spec

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Unified text + multimodal generation (Gemini-compatible signature)
    # ------------------------------------------------------------------
    def _generate(
        self,
        *,
        # ① legacy chat/completion entry-point --------------------------
        messages: List[Message] | None = None,
        # ② convenience keywords ---------------------------------------
        user_prompt:     str | None            = None,
        attachment_names: List[str] | None     = None,
        system_prompt:    str | None           = None,
        # ---------------------------------------------------------------
        stream: bool = False,
        **params: Any,
    ) -> Union[Dict[str, Any], Iterator[str]]:

        # -------- assemble messages list if caller used shortcuts -----
        if messages is None:
            if user_prompt is None:
                raise ValueError(
                    "HTTPChatModel.generate_content needs `user_prompt` "
                    "when `messages` is omitted."
                )

            msg_parts: List[dict] = []

            ## Attachments not implemented in Ollama, etc
            # if attachment_names:
            #     raise NotImplementedError(
            #     print(
            #         "This HTTP endpoint does not accept image parts. "
            #         "Remove `attachment_names` or use a multimodal-capable server."
            #     )            # encode attachments (URLs are fine; local files → base64 URI)
            # attachment_names = attachment_names or []
            # for ref in attachment_names:
            #     msg_parts.append(self._part_from_ref(ref))

            # user text
            msg_parts.append({"type": "text", "text": user_prompt})

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # OpenAI multimodal messages use list-of-parts as `content`
            messages.append({"role": "user", "content": msg_parts})

        payload: Dict[str, Any] = {
            "model":    self.model_name,
            "messages": messages,
            "stream":   stream,
            **params,
        }
        if self._functions:
            payload["tools"] = [
                {"type": "function", "function": spec} for spec in self._functions
            ]

        url  = f"{self.base_url}/chat/completions"
        resp = self.session.post(url, json=payload, stream=stream)
        resp.raise_for_status()

        if not stream:
            return resp.json()

        def _sse_text() -> Iterator[str]:
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                delta = chunk["choices"][0].get("delta", {})
                if delta.get("content") is not None:
                    yield delta["content"]
        return _sse_text()

    # ------------------------------------------------------------------
    # helper: turn a file-path or URL into an OpenAI image part
    # ------------------------------------------------------------------
    @staticmethod
    def _part_from_ref(ref: str) -> dict:
        """
        Returns a dict in the format expected by OpenAI-style multimodal APIs:
          {"type": "image_url", "image_url": {"url": "<...>"}}
        Local files are read & base64-encoded into a data-URI.
        """
        from urllib.parse import urlparse
        import base64, mimetypes, requests, pathlib

        # URL? just pass through
        if urlparse(ref).scheme in ("http", "https"):
            return {"type": "image_url", "image_url": {"url": ref}}

        # otherwise treat as local file path
        path = pathlib.Path(ref).expanduser()
        mime, _ = mimetypes.guess_type(path)
        with path.open("rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        data_uri = f"data:{mime or 'application/octet-stream'};base64,{b64}"
        return {"type": "image_url", "image_url": {"url": data_uri}}


    # ------------------------------------------------------------------
    # Tool-call extraction (handles both `function_call` and `tool_calls`)
    # ------------------------------------------------------------------
    def _iter_tool_calls(
        self, response: Dict[str, Any]
    ) -> List[Tuple[str, Dict[str, Any], Any]]:
        calls: List[Tuple[str, Dict[str, Any], Any]] = []
        msg = response.get("choices", [{}])[0].get("message", {})

        # # Legacy OpenAI style (some servers still emit this)
        # if (fc := msg.get("function_call")):
        #     args = self._parse_args(fc.get("arguments", {}))
        #     calls.append((fc["name"], args, fc))

        # New style: array of tool_calls
        for tc in msg.get("tool_calls", []):
            fn  = tc.get("function", {})
            name, args_raw = fn.get("name"), fn.get("arguments", {})
            if name:
                calls.append((name, self._parse_args(args_raw), tc))

        return calls

    @staticmethod
    def _parse_args(args_raw: Any) -> Dict[str, Any]:
        if isinstance(args_raw, str):
            try:   return json.loads(args_raw)
            except json.JSONDecodeError: return {}
        return args_raw if isinstance(args_raw, dict) else {}

    # ------------------------------------------------------------------
    def _response_text(self, response: Dict[str, Any]) -> str:
        return (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            or ""
        )


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
