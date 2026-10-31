"""Serve a local Muse Glimmer checkpoint for tool-use conversations."""

import argparse
import json
import re
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from siliconshield.research_loop.controller.inference_metrics import (
    GenerationStreamer,
    TokenTiming,
)


def parse_completion(text, tools):
    schemas = {
        t["function"]["name"]: t["function"]["parameters"] for t in tools
    }
    calls = []
    for name, body in re.findall(
        r'<atem:invoke name="([^"]+)">(.*?)</atem:invoke>', text, re.DOTALL
    ):
        if name not in schemas:
            raise ValueError("Model requested an unknown tool: " + name)
        parameters = {}
        properties = schemas[name].get("properties", {})
        for key, value in re.findall(
            r'<atem:parameter name="([^"]+)">(.*?)</atem:parameter>',
            body,
            re.DOTALL,
        ):
            if key not in properties or key in parameters:
                raise ValueError("Unknown or duplicate tool parameter")
            kind = properties[key].get("type")
            parameters[key] = value if kind == "string" else json.loads(value)
        if not set(schemas[name].get("required", [])).issubset(parameters):
            raise ValueError("Tool call is missing required arguments")
        calls.append(
            {
                "id": f"call_{len(calls)}_{time.time_ns()}",
                "type": "function",
                "function": {"name": name, "arguments": parameters},
            }
        )
    reasoning = re.findall(
        r"assistant to=self<\|message\|>(.*?)(?:<\|eom\|>|<\|eot\|>)",
        text,
        re.DOTALL,
    )
    answers = re.findall(
        r"assistant to=user<\|message\|>(.*?)(?:<\|eot\|>|$)", text, re.DOTALL
    )
    message = {"role": "assistant", "content": "\n".join(answers)}
    if calls:
        message["tool_calls"] = calls
    if reasoning:
        message["reasoning_content"] = "\n".join(reasoning)
    if not calls and not answers:
        raise ValueError(
            "Completion contains neither a valid tool call nor a final user response"
        )
    return message


def local_completion(model, tokenizer, payload, *, name, max_context=32768):
    import torch

    if payload.get("model") != name:
        raise ValueError("Request model must match the server --name")
    tools = payload.get("tools", [])
    messages = json.loads(json.dumps(payload["messages"]))
    for message in messages:
        for call in message.get("tool_calls", []):
            if isinstance(call["function"]["arguments"], str):
                call["function"]["arguments"] = json.loads(
                    call["function"]["arguments"]
                )
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        reasoning_strength="high",
    )
    inputs = tokenizer(
        rendered, add_special_tokens=False, return_tensors="pt"
    ).to("cuda")
    count = inputs["input_ids"].shape[1]
    limit = int(payload.get("max_tokens", 2048))
    if limit < 1 or count + limit > max_context:
        raise ValueError(
            "Requested output tokens must be positive and fit with the input within --max-context"
        )
    eot = tokenizer.convert_tokens_to_ids("<|eot|>")
    timing = TokenTiming(time.perf_counter())
    generation_options = {}
    if payload.get("record_timing", False):
        generation_options["streamer"] = GenerationStreamer(timing)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=limit,
            eos_token_id=eot,
            pad_token_id=eot,
            use_cache=True,
            **generation_options,
        )
    tokens = generated[0, count:]
    text = "<|start|>assistant" + tokenizer.decode(
        tokens, skip_special_tokens=False
    )
    if len(tokens) == limit and int(tokens[-1]) != eot:
        message, finish = {"role": "assistant", "content": text}, "length"
    else:
        message = parse_completion(text, tools)
        finish = "tool_calls" if message.get("tool_calls") else "stop"
    result = {
        "model": name,
        "choices": [{"message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": count, "completion_tokens": len(tokens)},
    }
    if generation_options:
        if timing.generated_tokens != len(tokens):
            raise RuntimeError("Generation timing token count is inconsistent")
        result["timing"] = timing.summary(time.perf_counter())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Path to the local base-model checkpoint"
    )
    parser.add_argument(
        "--adapter",
        help="LoRA checkpoint to load; omit to serve the base model",
    )
    parser.add_argument(
        "--name",
        default="muse-glimmer-controller",
        help="Model name clients must send in requests",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=18320, help="HTTP port (default: 18320)"
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=32768,
        help="Maximum input and output tokens combined",
    )
    args = parser.parse_args()
    from siliconshield.research_loop.execution.gpu_lease import acquire_gpu

    gpu_lease = acquire_gpu()
    if gpu_lease.closed:
        raise RuntimeError(
            "Glimmer GPU lease is not held; release the other holder "
            "(training, serving, or evaluation) and retry"
        )
    import os

    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, MuseGlimmerForConditionalGeneration

    from siliconshield.training.glimmer.identity import (
        MODEL_REVISION,
        verify_snapshot,
    )

    verify_snapshot(
        args.model,
        os.environ.get("SILICONSHIELD_MODEL_FINGERPRINT"),
        revision=MODEL_REVISION,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = MuseGlimmerForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="sdpa",
        local_files_only=True,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self.respond(
                    200,
                    {
                        "ready": True,
                        "model": args.name,
                        "adapter": bool(args.adapter),
                    },
                )
            else:
                self.respond(404, {"error": "Unknown route"})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.respond(404, {"error": "Unknown route"})
                return
            try:
                payload = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
                self.respond(
                    200,
                    local_completion(
                        model,
                        tokenizer,
                        payload,
                        name=args.name,
                        max_context=args.max_context,
                    ),
                )
            except Exception as exc:
                # Endpoint isolation: a bad request gets a 400 while the
                # server keeps serving subsequent requests.
                self.respond(
                    400, {"error": type(exc).__name__, "detail": str(exc)}
                )

    print(
        json.dumps({"phase": "serving", "model": args.name, "port": args.port}),
        flush=True,
    )
    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
