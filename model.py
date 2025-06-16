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
* ``response_text``       - extract printable text
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
  try:
    with open("gemini.api_key") as file:
        os.environ["GEMINI_API_KEY"] = file.read()
  except:
    pass

###############################################################################
# Utility
###############################################################################

class RateLimitTracker:
    """Throttle helper for free-tier endpoints."""

    def __init__(self, per_min_limit: int):
        self.per_min_limit = per_min_limit
        self.history: List[datetime] = []

    def log_query(self) -> None:
        now = datetime.now()
        self.history.append(now)
        self.history = [t for t in self.history if (now - t).seconds <= 60]

    def print(self):
        print(self.history)

    def time_to_wait(self) -> float:
        if self.per_min_limit < 0:
            return 0
        average_wait = 60 / self.per_min_limit
        n = len(self.history)
        # Spend the first quarter of the budget immediately
        # Spend the second quarter of the budget a bit quickly
        # Spend spend the second half slowly enough to avoid rate limiting
        if n < self.per_min_limit // 4:
            return 0.0
        if n < self.per_min_limit // 2:
            return average_wait / 2
        return average_wait * 2
    
    def wait(self):
        self.log_query()
        wait = self.time_to_wait()
        if wait > 0:
            time.sleep(wait)
        


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
        *,
        verbose: bool = False,
        **kwargs
    ) -> None:
        self.model_name = model_name
        self.rate = RateLimitTracker(per_min_limit)
        self.per_day_limit = per_day_limit
        self.tool_registry: Dict[str, Dict[str, Any]] = {}
        self.verbose = verbose
        self.native_tool = kwargs["native_tool"] if "native_tool" in kwargs else True

        self.allow_system_prompt = kwargs["allow_system_prompt"] if "allow_system_prompt" in kwargs else True

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def generate_content(self,
                         *, # args below will be displayed as hints
                         user_prompt : str |  None = None,
                         system_prompt : str |  None = None,
                         history : List[dict] |  None = None,
                         **kwargs):

        # append expected args to kwargs for processing
        if user_prompt:         
            kwargs["user_prompt"] = user_prompt
        if system_prompt:
            kwargs["system_prompt"] = system_prompt
        if history:
            kwargs["history"] = history

        self.rate.log_query()
        wait = self.rate.time_to_wait()
        if wait:
            time.sleep(wait)

        # Handle non-native tool calling by embedding description into system prompt
        if not self.native_tool and "tools" in kwargs: 
            if self.verbose: print("Processing non-native tools")

            tools_provided = kwargs["tools"]
            # Embed tool instructions in system prompt
            # 1. Build tool description block
            tool_persona = "You have access to the following tools."
            tools_block = yaml.dump(tools_provided)
            # tools_block = []
            # for spec in self._tools:
                # tools_block.append(
                #     f"- {spec['name']} :: {spec['description']}  "
                #     f"params = {json.dumps(spec['parameters']['properties'])}"
                # )
            guard = (
                "You may call **one** function. "
                "If you do, respond with *only* this JSON:\n"
                '{ "function": <func_name>, "arguments": {<arg_name_1>:<arg_val_1>, <arg_name_2>:<arg_val_2>,...}}'
            )
            tool_prompt  = f"{tool_persona}\nTOOLS:\n{tools_block}\n{guard}"
            if "messages" in kwargs:
                print("Warning: tool instruction not added to pre-existing messages")

            if "system_prompt" in kwargs:
                kwargs["system_prompt"] += tool_prompt
            else:
                kwargs["system_prompt"] = tool_prompt

            if self.verbose: print(kwargs["system_prompt"])
        

        # Handle models that do not allow system prompts
        if not self.allow_system_prompt and "system_prompt" in kwargs:
            print("Processing forbidden system prompt")
            system_prompt = kwargs.pop("system_prompt")
            user_prompt = kwargs["user_prompt"] if "user_prompt" else ""
            kwargs["user_prompt"] = "BEGIN SYSTEM PROMPT\n" \
                                    + system_prompt         \
                                    + "\nEND SYSTEM PROMPT\n\nBEGIN USER PROMPT\n" \
                                    + user_prompt           \
                                    + "\n END USER PROMPT"
        if not self.allow_system_prompt and "messages" in kwargs:
            print("WARNING: not fixing system prompts in messages list")

        if self.verbose:
            print("kwargs")
            pprint.pp(kwargs)

        # Call subclass's version of llm query
        try:
            content = self._generate(verbose=self.verbose,**kwargs)
        except Exception as e:
            print("Error generating content:" , e)
            pprint.pp(kwargs)
            content = f"Error: {e}"

        # Update history
        if history is not None:
            if system_prompt:
                history.append(
                    {
                        "role" : "system",
                        "content" : kwargs["system_prompt"]
                    }
                )
            if user_prompt:
                history.append(
                    {
                        "role" : "user",
                        "content" : kwargs["user_prompt"]
                    }
                )
            tool_calls = self._iter_tool_calls(content)
            if tool_calls is not None:
                for call in tool_calls:
                    history.append(
                            {
                                "role" : "function",
                                "content" : call[2]
                            }
                    )
            else:
                history.append(
                    {
                        "role" : "assistant",
                        "content" : self.response_text(content)
                    }
                )

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
        if name in self.tool_registry:
            print(f"Warning: Tool {name} already registered. Using PREVIOUS definition.")
            return self.tool_registry[name]["tool"]
        # call subclass appropriate function to create the tool object itself
        tool_obj = self._create_tool_object(name, description, parameters)
        #
        self.tool_registry[name] = {
            "function": func, 
            "tool": tool_obj, 
            "description": description, 
            "parameters": parameters}
        return tool_obj

    def get_tool(self, name: str):
        if name not in self.tool_registry:
            print(f"Error: tool {name} is not registered")
        return self.tool_registry.get(name, {}).get("tool")

    # .................................................................
    def apply_tool(self, response, **kwargs):
        # Use subclass's function to return iterable of tool calls
        calls = self._iter_tool_calls(response)
        if not calls:
            return [("No Tool Calls", "")]
        results: List[Tuple[Any, Any]] = []
        for name, args, raw in calls:
            # Pass both the LLM supplied arguments AND the user defined **kwargs 
            merged = {**(args or {}), **(kwargs or {})}
            if name in self.tool_registry:
                try:    
                    out = self.tool_registry[name]["function"](**merged)
                    results.append((out, raw))
                except Exception as e:
                    print(self.tool_registry[name])
                    # raise ValueError("Error applying tool", {name}, "\n", raw, "\n", e)
                    print(f"=========\n ERROR applying tool {name}.\nIn: {raw}\nError{e}\n=========")
                    return [("__error__", {}, f"tool call must be valid json: {e}")]                    
                    # results.append((f"Error applying tool {raw}.\n Tool Call: ", e))
            else:
                results.append((f"Tool {name} not registered", raw))
        return results

    def print_response(self, response):
        calls = self._iter_tool_calls(response)
        if calls:
            print([raw for *_unused, raw in calls])
        else:
            print(self.response_text(response))

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
    def response_text(self, response) -> str:
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
        **opts, # Optional arguments (e.g. verbose) to pass to SimpleModel 
    ) -> None:
        super().__init__(model_name, per_min_limit, per_day_limit, **opts)
        # if genai is None:
        #     raise ImportError("google-genai not installed - `pip install google-genai`.")
        self.API_KEY = os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client( api_key=self.API_KEY )


    # Provider hooks ----------------------------------------------------
    ## NOTE: A lone * in a function definition indicates that
    #        all parameters after it must be passed as keyword 
    #        arguments
    # BUT: Here they have default values of None, so they are actually optional?
    def _generate(
        self,
        *, # Parameters below "*," are optional because they have default values.  Otherwise, they're required
        user_prompt: str | None = None,
        history: List[dict] | None = None,
        # contents: List[gtypes.Content] | None = None,
        tools:    List[Any]           | None = None,
        attachment_names: List[str]   | None = None,
        system_prompt:    str         | None = None,
        **kwargs,
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
        contents_list = []
        attachment_names = attachment_names or []

        # 0. Read in previous history
        if history:
            for msg in history:
                contents_list.append(
                    gtypes.Content(
                        role = msg["role"],
                        parts = [gtypes.Part.from_text(text=msg["content"])]
                    )
                )

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


        if "native_tool" in kwargs:
            if not kwargs["native_tool"]:
                tools=None

        # ------------------------------------------------------------------
        if self.native_tool:
            cfg = gtypes.GenerateContentConfig(
                tools=tools,
                response_mime_type="text/plain",
            )
        else:
            cfg = gtypes.GenerateContentConfig(
                response_mime_type="text/plain",
            )

        if self.verbose:
            pprint.pp(contents_list)

        return self.client.models.generate_content(
            model=self.model_name,
            contents=contents_list,
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
        if calls is not None:
            return [(c.name, c.args, c) for c in calls]
        else:
            return None

    def response_text(self, response):
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
            return gtypes.Part.from_bytes(data=fh.read(), mime_type=str(mime) )

###############################################################################
# HTTP based access 
###############################################################################
import json, requests
from typing import Any, Dict, List, Iterator, Tuple, Union

Message = Dict[str, Union[str, list, dict]]            # helper alias

class HTTPChatModel(SimpleModel):
    """
    Generic client for any OpenAI-compatible HTTP server (vLLM, Ollama, LocalAI …).
    """

    def __init__(
        self,
        model_name: str,
        base_url: str                    = "http://localhost:11434",
        api_key:   str | None            = None,
        *, # Parameters below "*," are optional because they have default values.  Otherwise, they're required
        native_tool : bool = True,
        multimodal : bool = False,
        **opts, # Additional (optional) arguments show up here. E.g. to pass verbose to SimpleModel wrapper
    ) -> None:
        super().__init__(model_name, **opts)
        self.base_url    = base_url.rstrip("/")
        self.api_key     = api_key or ""

        # Set behavior flags
        self.native_tool = native_tool
        self.multimodal = multimodal

        # Create the session for communicating with the LLM
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
        spec: Dict[str, Any] = {
            "name":        name,
            "description": description,
            "parameters": {
                "type":       "object",
                "properties": parameters,
                "required" : list(parameters.keys()),
            },
        }
        return spec

    # ------------------------------------------------------------------
    # Unified text + multimodal generation 
    # ------------------------------------------------------------------
    def _generate(
        self,
        *,
        # convenience keywords ---------------------------------------
        user_prompt:     str | None            = None,
        system_prompt:    str | None           = None,
        history: List[dict] | None             = None,
        attachment_names: List[str] | None     = None,
        # ---------------------------------------------------------------
        stream: bool = False,
        **params: Any,
    ) -> Union[Dict[str, Any], Iterator[str]]:

        # Initialize object we will send to the model
        payload: Dict[str, Any] = {
            "model":    self.model_name,
            "stream":   stream,
        }

        if "tools" in params:
            tools_used = params["tools"]
            if self.verbose: print(tools_used)

        # Handle native tools
        # Non-native tool calling already handled by generate_content in SimpleModel 
        if "tools" in params and self.native_tool:
            payload["tools"] = [
                {"type": "function", "function": spec} for spec in tools_used
            ]

        # -------- assemble messages list if caller used shortcuts -----
        if user_prompt is None:
            raise ValueError(
                "HTTPChatModel.generate_content needs `user_prompt` "
                "when `messages` is omitted."
            )

        messages = []

        if history:
            messages.extend(history)

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if self.multimodal and attachment_names:
            parts = [{"type": "text", "text": user_prompt}]
            for ref in attachment_names:
                parts.append(self._part_from_ref(ref))
            messages.append({"role": "user", "content": parts})
        else:
            # Non multimodal models expect text content only (no parts)
            messages.append({"role": "user", "content": user_prompt})

        payload["messages"] = messages

        if self.verbose:
            print("LLM Query JSON:")
            pprint.pp(payload)

        url  = f"{self.base_url}/v1/chat/completions"
        resp = self.session.post(url, json=payload, stream=stream)
        # resp.raise_for_status()
        resp = self.session.post(url, json=payload, stream=stream)

        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            print("Server returned ERROR: ", resp.status_code)
            try:
                pprint.pp( json.loads(resp.text) )
            except:
                print( resp.text )
            # raise
            return {"error" : e}

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

        # For local models, must download and convert images to base64
        def to_data_uri(binary: bytes, mime: str) -> str:
            b64 = base64.b64encode(binary).decode()
            return f"data:{mime};base64,{b64}"
    
        # download and process url
        if urlparse(ref).scheme in ("http", "https"):
            # return {"type": "image_url", "image_url": {"url": ref}}

            resp = requests.get(ref, timeout=10)
            resp.raise_for_status()
            mime = resp.headers.get("Content-Type") or mimetypes.guess_type(ref)[0]
            if not (mime or "").startswith("image/"):
                raise ValueError(f"URL does not point to an image (mime={mime})")
            data_uri = to_data_uri(resp.content, mime or "image/jpeg")
            return {"type": "image_url", "image_url": {"url": data_uri}}
 
        # otherwise treat as local file path
        path = pathlib.Path(ref).expanduser()
        mime, _ = mimetypes.guess_type(path)
        with path.open("rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        # data_uri = f"data:{mime or 'application/octet-stream'};base64,{b64}"
        data_uri = to_data_uri(path.read_bytes(), mime or "image/jpeg")
        return {"type": "image_url", "image_url": {"url": data_uri}}


    # ------------------------------------------------------------------
    # Tool-call extraction (handles both `function_call` and `tool_calls`)
    # ------------------------------------------------------------------
    def _iter_tool_calls(
        self, response: Dict[str, Any] | str
    ) -> List[Tuple[str, Dict[str, Any], Any]]:
        calls: List[Tuple[str, Dict[str, Any], Any]] = []

        if self.verbose: print("Parsing response for tool calls.  Response: ", response )

        if self.native_tool:
            print("Response",response)
            msg = response.get("choices", [{}])[0].get("message", {})

            # Recieve calls from OpenAPI  
            for tc in msg.get("tool_calls", []):
                fn  = tc.get("function", {})
                name, args_raw = fn.get("name"), fn.get("arguments", {})

                if isinstance(args_raw, str):
                    try:   args = json.loads(args_raw)
                    except json.JSONDecodeError: args = {}

                # if name:
                calls.append((name, args, tc))
                # print(tc)
                # print(name, args_raw)

            return calls
        
        else: 
            try:
                # pull the text between the first and last bracket
                # ASSUMES the model outputs a SINGLE tool call
                response_message = self.response_text(response)
                first = response_message.index("{")
                last  = response_message.rindex("}") + 1
                tool_json = json.loads(response_message[first:last])
                # Extract function and arguments from json
                function_name = tool_json.get("function")
                args = tool_json.get("arguments")
                if self.verbose:
                    print(function_name)
                    pprint.pp(args)
                    pprint.pp(response)
                # Extract calls using regexp
                # Extract name and arguments from call
                # Return in appropriate format
                # print("Error: Non-native tool calling not yet implemented")
                return [(function_name,args,response)]
            except:
                return[("Error: tool call must be valid json of correct format",{},response_message)]

    # ------------------------------------------------------------------
    def response_text(self, response: Dict[str, Any]) -> str:
        try:
            return (
                response.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                or ""
            )
        except:
            # print("ERROR parsing response text.  Full response:", str)
            return str(response)

###############################################################################
# Hugging Face Transformers  (transformers ≥ 4.41 recommended)
###############################################################################
###############################################################################
# Hugging Face Transformers (transformers ≥ 4.41)
###############################################################################
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        PreTrainedTokenizerBase,
    )
    import torch, re as _re, json as _json
    if hasattr(torch, "_dynamo"):
        torch._dynamo.config.suppress_errors = True
except ModuleNotFoundError:                                # library optional
    AutoTokenizer = AutoModelForCausalLM = PreTrainedTokenizerBase = None  # type: ignore


class HFTransformersModel(SimpleModel):
    """
    Wrapper for chat-template-aware Hugging Face models (Zephyr-β, Llama-3-
    Instruct, Phi-3-mini-chat, …).  When *native_tool=True* we hand the tool
    schema to `tokenizer.apply_chat_template`; otherwise the base class embeds
    tool instructions in the system prompt.
    """

    # ――― tiny “call” object so `apply_tool` works the same way ―――
    class _ToolCall:
        def __init__(self, name: str, args: dict):
            self.name, self.args = name, args

    # ------------------------------------------------------------------
    def __init__(
        self,
        model_name: str = "microsoft/Phi-4-mini-reasoning",
        *,
        device: str | None = None,      # "cuda", "mps", "cpu", or None→auto
        dtype:  str       = "auto",     # "float16", "bfloat16", "auto", …
        per_min_limit: int = 30,
        **opts,                         # forwarded to SimpleModel
    ) -> None:
        super().__init__(model_name, per_min_limit, **opts)

        if AutoTokenizer is None:
            raise ImportError("`transformers` not installed – pip install transformers")

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name, padding_side="left"
        )
        torch_dtype = getattr(torch, dtype) if dtype != "auto" and hasattr(torch, dtype) else None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device or "auto",
        )
        self.model.eval()

    # ======================= abstract-hook implementations ===================
    # 1) tool schema helper ---------------------------------------------------
    def _create_tool_object(self, name: str, description: str, parameters: dict) -> dict:
        return {
            "type": "function",
            "function": {
                "name":        name,
                "description": description,
                "parameters": {
                    "type":       "object",
                    "properties": parameters,
                    "required":   list(parameters.keys()),
                },
            },
        }

    # 2) main generation routine ---------------------------------------------
    def _generate(
        self,
        *,
        user_prompt:   str | None           = None,
        system_prompt: str | None           = None,
        history:       str | None           = None,
        tools:         List[dict] | None    = None,
        max_new_tokens: int = 256,
        temperature:    float | None        = None,
        **gen_kw,
    ):
        # ---- normalise to a messages list --------------------------------
        if user_prompt is None:
            raise ValueError("HFTransformersModel needs `user_prompt`.")
        messages = []

        if history:
            messages.extend(history)

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        # ---- inject JSON schema when native_tool=True --------------------
        schema_for_template = tools if (self.native_tool and tools) else None

        enc_dict = self.tokenizer.apply_chat_template(
            messages,
            tools=schema_for_template,
            add_generation_prompt=True,
            return_dict=True,        # ⇐ ensures we get a **mapping**
            return_tensors="pt",
        ).to(self.model.device)

        # enc_dict is now a mapping: {"input_ids": Tensor, "attention_mask": Tensor}
        # with torch.no_grad():
        with torch.inference_mode():
            gen_ids = self.model.generate(
                **enc_dict,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                # **gen_kw,
            )

        reply = self.tokenizer.decode(
            gen_ids[0][enc_dict["input_ids"].shape[1] :],
            skip_special_tokens=False,
        )

        # wrap in a tiny object understood by apply_tool/print_response
        # return type("HFResp", (), {"text": reply})()
        return reply

    # 3) tool-call iterator ---------------------------------------------------
    def _iter_tool_calls(self, response) -> List[Tuple[str, Dict[str, Any], Any]] | None:
        # text = getattr(response, "text", str(response))
        text = response
        calls: List[Tuple[str, Dict[str, Any], Any]]  = []

        if self.native_tool:
            # a) native tool-calling: model emits <tool_call>{…}</tool_call>
            for m in _re.finditer(r"<tool_call>(.*?)</tool_call>", text, flags=_re.S):
                try:
                    obj = _json.loads(m.group(1))
                    if {"name", "arguments"} <= obj.keys():
                        calls.append((obj["name"], obj["arguments"], m.group(0)))
                except Exception:
                    return None
        else:
                try:
                    # pull the text between the first and last bracket
                    # ASSUMES the model outputs a SINGLE tool call
                    response_message = self.response_text(response)
                    first = response_message.index("{")
                    last  = response_message.rindex("}") + 1
                    tool_json = json.loads(response_message[first:last])
                    # Extract function and arguments from json
                    function_name = tool_json.get("function")
                    args = tool_json.get("arguments")
                    if self.verbose:
                        print(function_name)
                        pprint.pp(args)
                        pprint.pp(response)
                    # Extract calls using regexp
                    # Extract name and arguments from call
                    # Return in appropriate format
                    # print("Error: Non-native tool calling not yet implemented")
                    calls = [(function_name,args,response)]            # # b) fallback: any bare JSON with those keys
                # if not calls:
                #     for m in _re.finditer(r"\\{[^\\}]*\\}", text):
                #         try:
                #             obj = _json.loads(m.group(0))
                #             if {"name", "arguments"} <= obj.keys():
                #                 calls.append((obj["name"], obj["arguments"], m.group(0)))
                #         except Exception:
                #             continue
                except: 
                    return None
        return calls

    # 4) extract human-readable answer ---------------------------------------
    def response_text(self, response) -> str:
        # strip out everything formatted like a special token <| ... |>
        clean = _re.sub(r'<\|.*?\|>', '', response)
        return clean


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
    # self.verbose=True

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
