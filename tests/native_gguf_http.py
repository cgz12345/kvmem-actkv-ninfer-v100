"""Opt-in native GGUF serving checks; server lifecycle is managed by the caller."""
import argparse
import base64
import json
import struct
import time
import urllib.request
import zlib
from pathlib import Path


def png(size):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    pixels = (b"\0" + b"\xff\0\0" * size) * size
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))


def red_video(size=256, frames=4):
    """Uncompressed AVI fixture with real temporal frames, no external encoder."""
    def chunk(kind, data):
        return kind + struct.pack("<I", len(data)) + data + (b"\0" if len(data) % 2 else b"")
    def listed(kind, data):
        return chunk(b"LIST", kind + data)
    frame = b"\0\0\xff" * (size * size)
    avih = struct.pack("<14I", 500000, len(frame) * 2, 0, 16, frames, 0, 1,
                       len(frame), size, size, 0, 0, 0, 0)
    strh = struct.pack("<4s4sIHH8I4h", b"vids", b"DIB ", 0, 0, 0, 0, 1, 2, 0,
                       frames, len(frame), 0xffffffff, 0, 0, 0, size, size)
    strf = struct.pack("<IiiHHIIiiII", 40, size, size, 1, 24, 0, len(frame), 0, 0, 0, 0)
    header = listed(b"hdrl", chunk(b"avih", avih) + listed(b"strl", chunk(b"strh", strh) + chunk(b"strf", strf)))
    movie = listed(b"movi", chunk(b"00db", frame) * frames)
    index = chunk(b"idx1", b"".join(struct.pack("<4sIII", b"00db", 16,
                  4 + i * (len(frame) + 8), len(frame)) for i in range(frames)))
    body = b"AVI " + header + movie + index
    return b"RIFF" + struct.pack("<I", len(body)) + body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8089")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = {}

    def call(path, body=None):
        request = urllib.request.Request(args.url + path,
            None if body is None else json.dumps(body, ensure_ascii=False).encode(),
            {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=900) as response:
            return json.load(response)

    def save(name, result):
        results[name] = result
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(name, "PASS", result.get("usage", {}), flush=True)

    for attempt in range(120):
        try:
            if call("/health")["status"] == "ok":
                break
        except Exception:
            time.sleep(1)
    else:
        raise RuntimeError("server did not become ready")
    save("models", call("/v1/models"))
    base = {"model": "qwen3.8-27b", "temperature": 0, "max_tokens": 512,
            "chat_template_kwargs": {"enable_thinking": False}}
    result = call("/v1/chat/completions", {**base, "messages": [{"role": "user", "content": "计算17乘23，只回答结果。"}]})
    assert "391" in result["choices"][0]["message"]["content"], result
    save("math", result)
    tools = [{"type": "function", "function": {"name": "record_result", "description": "记录计算结果",
        "parameters": {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]}}}]
    result = call("/v1/chat/completions", {**base, "tools": tools,
        "messages": [{"role": "user", "content": "计算17乘23，用record_result记录结果。"}]})
    choice = result["choices"][0]
    assert choice["finish_reason"] == "tool_calls", result
    tool = choice["message"]["tool_calls"][0]["function"]
    assert tool["name"] == "record_result" and json.loads(tool["arguments"])["value"] == 391, result
    save("tools", result)
    image = "data:image/png;base64," + base64.b64encode(png(1024)).decode()
    for detail in ("high", "low"):
        result = call("/v1/chat/completions", {**base,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "What is the main color? Answer with one English word."},
                {"type": "image_url", "image_url": {"url": image, "detail": detail}}]}]})
        assert "red" in result["choices"][0]["message"]["content"].lower(), result
        save("image_" + detail, result)
    assert results["image_low"]["usage"]["prompt_tokens"] < results["image_high"]["usage"]["prompt_tokens"]
    result = call("/v1/responses", {"model": base["model"], "temperature": 0, "max_output_tokens": 512,
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": "What is the main color? Answer with one English word."},
            {"type": "input_image", "image_url": image, "detail": "low"}]}]})
    assert "red" in json.dumps(result.get("output", []), ensure_ascii=False).lower(), result
    save("responses_image", result)
    video = "data:video/x-msvideo;base64," + base64.b64encode(red_video()).decode()
    result = call("/v1/chat/completions", {**base,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "What main color appears in this video? Answer with one English word."},
            {"type": "video_url", "video_url": video}]}]})
    assert "red" in result["choices"][0]["message"]["content"].lower(), result
    save("video", result)
    request = urllib.request.Request(args.url + "/v1/chat/completions", json.dumps({**base,
        "stream": True, "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": "计算17乘23，只回答结果。"}]}).encode(),
        {"Content-Type": "application/json"})
    chunks = []
    done = False
    with urllib.request.urlopen(request, timeout=900) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                done = True
                break
            chunks.append(json.loads(payload))
    content = "".join(c.get("delta", {}).get("content", "")
                      for item in chunks for c in item.get("choices", []))
    assert done and "391" in content and any(item.get("usage") for item in chunks), chunks
    save("stream", {"content": content, "chunks": chunks})
    # A prompt beyond the 20K/32K GPU window exercises KVMem selection and replay.
    prompt = ("背景记录：系统正常，无需采取行动。\n" * 2000
              + "唯一启动码是 ZX-7749-QK，请记住。\n"
              + "背景记录：系统正常，无需采取行动。\n" * 2000
              + "唯一启动码是什么？只回答代码。")
    result = call("/v1/responses", {"model": base["model"], "temperature": 0,
        "max_output_tokens": 512, "input": prompt})
    assert result["usage"]["input_tokens"] > 32768, result["usage"]
    assert "ZX-7749-QK" in json.dumps(result.get("output", [])), result
    save("long", result)
    follow = call("/v1/responses", {"model": base["model"], "temperature": 0,
        "max_output_tokens": 512, "previous_response_id": result["id"],
        "input": "再说一次唯一启动码，只回答代码。"})
    assert "ZX-7749-QK" in json.dumps(follow.get("output", [])), follow
    assert follow["usage"]["input_tokens_details"]["cached_tokens"] > 0, follow["usage"]
    save("reuse", follow)
    save("health", call("/health"))
    save("stats", call("/stats"))
    print("All native GGUF HTTP checks passed", flush=True)


if __name__ == "__main__":
    main()
