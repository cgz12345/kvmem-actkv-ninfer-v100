"""Agent-compatibility probes against a live server.

The clients that connect to this server (coding agents, chat UIs, SDKs) each use a
slightly different slice of the OpenAI Chat/Responses and Anthropic Messages wire
formats. This script exercises those slices against a running instance and fails
loudly when one of them breaks, so a release can be checked before it is handed out.

Usage:
    python tests/serve_agent_compat.py --base http://127.0.0.1:8088
    python tests/serve_agent_compat.py --base http://127.0.0.1:8088 --model qwen3.8-27b

It only sends small requests. A tool-capable model is assumed; probes are written to
pass on either "tool call" or "plain answer" where both are legitimate.
Exit code 0 means every probe passed.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

PASS = 'PASS'
FAIL = 'FAIL'


class Probes:
    def __init__(self, base: str, model: str) -> None:
        self.base = base.rstrip('/')
        self.model = model
        self.failures = 0
        self.results: list[tuple[str, str, str]] = []

    def record(self, name: str, ok: bool, detail: str = '') -> None:
        self.results.append((name, PASS if ok else FAIL, detail))
        if not ok:
            self.failures += 1
        print(f'{name:44s} {PASS if ok else FAIL}  {detail}')

    def request(self, path: str, body: dict | None = None, method: str = 'POST',
                headers: dict | None = None, stream: bool = False):
        url = f'{self.base}{path}'
        data = json.dumps(body).encode('utf-8') if body is not None else None
        head = {'Content-Type': 'application/json; charset=utf-8'}
        if headers:
            head.update(headers)
        req = urllib.request.Request(url, data=data, headers=head, method=method)
        try:
            resp = urllib.request.urlopen(req, timeout=300)
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode('utf-8', 'replace')
        if stream:
            return resp.status, resp
        return resp.status, resp.read().decode('utf-8')

    def json_request(self, path: str, body: dict | None = None, headers: dict | None = None,
                     method: str = 'POST'):
        status, payload = self.request(path, body, headers=headers, method=method)
        try:
            return status, json.loads(payload) if isinstance(payload, str) else payload
        except json.JSONDecodeError:
            return status, {'_raw': payload[:200]}

    def chat(self, **fields) -> dict:
        body = {'model': self.model,
                'messages': [{'role': 'user', 'content': 'Reply with the single word: ok'}],
                'max_tokens': 32, 'reasoning_effort': 'none'}
        body.update(fields)
        return body

    # --- probes ---------------------------------------------------------------

    def probe_models(self) -> None:
        status, data = self.json_request('/v1/models', method='GET')
        ok = status == 200 and isinstance(data.get('data'), list) and data.get('data')
        self.record('GET /v1/models lists models', ok, f'http {status}')
        model_id = data['data'][0]['id'] if ok else self.model
        status, one = self.json_request(f'/v1/models/{model_id}', method='GET')
        self.record('GET /v1/models/{id} retrieves one',
                    status == 200 and one.get('id') == model_id, f'http {status}')

    def probe_cors_preflight(self) -> None:
        status, _ = self.request('/v1/chat/completions', None, method='OPTIONS', headers={
            'Origin': 'http://localhost:5173',
            'Access-Control-Request-Method': 'POST',
            'Access-Control-Request-Headers': 'authorization, content-type'})
        self.record('OPTIONS preflight for browser agents', 200 <= status < 300, f'http {status}')

    def probe_chat_basic(self) -> None:
        status, data = self.json_request('/v1/chat/completions', self.chat())
        ok = status == 200 and data.get('choices')
        self.record('chat: minimal request', ok, f'http {status}')

    def probe_chat_json_object(self) -> None:
        body = self.chat(messages=[{'role': 'user',
                                    'content': 'Return the object {"ok": true} as JSON.'}],
                         response_format={'type': 'json_object'})
        status, data = self.json_request('/v1/chat/completions', body)
        text = data.get('choices', [{}])[0].get('message', {}).get('content', '') if status == 200 else ''
        parses = False
        try:
            parses = isinstance(json.loads(text), dict)
        except (json.JSONDecodeError, TypeError):
            parses = False
        self.record('chat: response_format json_object accepted', status == 200,
                    f'http {status}, parses={parses}')
        self.record('chat: json_object answer is a JSON object', parses, text[:48])

    def probe_chat_controls(self) -> None:
        body = self.chat(stop=['gamma'], messages=[{'role': 'user',
                        'content': 'Say exactly: alpha beta gamma delta'}])
        status, data = self.json_request('/v1/chat/completions', body)
        text = data.get('choices', [{}])[0].get('message', {}).get('content', '') if status == 200 else ''
        self.record('chat: stop sequence halts output', status == 200 and 'gamma' not in text,
                    f'http {status}')
        status, _ = self.json_request('/v1/chat/completions', self.chat(max_completion_tokens=16))
        self.record('chat: max_completion_tokens alias', status == 200, f'http {status}')
        status, _ = self.json_request('/v1/chat/completions', {
            'model': self.model, 'max_tokens': 16, 'reasoning_effort': 'none',
            'messages': [{'role': 'developer', 'content': 'Be terse.'},
                         {'role': 'user',
                          'content': [{'type': 'text', 'text': 'Reply with the single word: ok'}]}]})
        self.record('chat: developer role + content parts', status == 200, f'http {status}')
        body = self.chat(future_unknown_field={'ignored': True})
        status, _ = self.json_request('/v1/chat/completions', body)
        self.record('chat: unknown top-level field ignored', status == 200, f'http {status}')

    def probe_chat_tools(self) -> None:
        tools = [{'type': 'function', 'function': {
            'name': 'get_weather', 'description': 'Look up weather',
            'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}},
                           'required': ['city']}}}]
        body = self.chat(tools=tools, tool_choice='auto',
                         messages=[{'role': 'user',
                                    'content': 'What is the weather in Paris? Use the tool.'}])
        status, data = self.json_request('/v1/chat/completions', body)
        called = bool(data.get('choices', [{}])[0].get('message', {}).get('tool_calls')) if status == 200 else False
        self.record('chat: tools with tool_choice auto', status == 200, f'http {status}, called={called}')
        # The engine cannot force one specific tool, so a named choice must be a clear 400 -
        # a client needs to know its request was not honored as written.
        status, data = self.json_request('/v1/chat/completions', self.chat(
            tools=tools, tool_choice={'type': 'function', 'function': {'name': 'get_weather'}}))
        self.record('chat: named tool_choice rejected explicitly', status == 400,
                    f'http {status}, code={data.get("error", {}).get("code")}')

    def probe_chat_stream(self) -> None:
        body = self.chat(stream=True, stream_options={'include_usage': True})
        status, resp = self.request('/v1/chat/completions', body, stream=True)
        saw_delta = saw_usage = saw_done = False
        if status == 200:
            for raw in resp:
                line = raw.decode('utf-8', 'replace').strip()
                if not line.startswith('data: '):
                    continue
                payload = line[6:]
                if payload == '[DONE]':
                    saw_done = True
                    break
                event = json.loads(payload)
                if event.get('choices') and event['choices'][0].get('delta', {}).get('content'):
                    saw_delta = True
                if event.get('usage'):
                    saw_usage = True
        self.record('chat: stream ends with [DONE]+usage',
                    status == 200 and saw_delta and saw_done and saw_usage,
                    f'http {status}, delta={saw_delta} usage={saw_usage} done={saw_done}')

    def probe_responses(self) -> None:
        status, data = self.json_request('/v1/responses', {
            'model': self.model, 'input': 'Reply with the single word: ok',
            'max_output_tokens': 32, 'reasoning': {'effort': 'none'}})
        ok = status == 200 and (data.get('output') or data.get('status') in ('completed', 'incomplete'))
        self.record('responses: minimal request', ok, f'http {status}')
        status, _ = self.json_request('/v1/responses', {
            'model': self.model, 'input': 'Reply with the single word: ok',
            'max_output_tokens': -1, 'reasoning': {'effort': 'none'}})
        self.record('responses: max_output_tokens -1 means no limit', status == 200, f'http {status}')
        status, _ = self.json_request('/v1/responses', {
            'model': self.model, 'instructions': 'Be terse.',
            'input': [{'role': 'user', 'content': [{'type': 'input_text', 'text': 'say ok'}]}],
            'max_output_tokens': 16, 'reasoning': {'effort': 'none'}})
        self.record('responses: instructions + typed input', status == 200, f'http {status}')

    def probe_anthropic(self) -> None:
        status, data = self.json_request('/v1/messages', {
            'model': self.model, 'max_tokens': 32,
            'system': [{'type': 'text', 'text': 'Be terse.',
                        'cache_control': {'type': 'ephemeral'}}],
            'messages': [{'role': 'user', 'content': 'Reply with the single word: ok'}]})
        ok = status == 200 and data.get('content')
        self.record('anthropic: system blocks + cache_control', ok, f'http {status}')
        status, data = self.json_request('/v1/messages/count_tokens', {
            'model': self.model,
            'messages': [{'role': 'user', 'content': 'Count these tokens please'}]})
        ok = status == 200 and isinstance(data.get('input_tokens'), int) and data['input_tokens'] > 0
        self.record('anthropic: count_tokens endpoint', ok, f'http {status}')
        status, _ = self.json_request('/v1/messages', {
            'model': self.model, 'max_tokens': 32, 'stop_sequences': ['gamma'],
            'messages': [{'role': 'user', 'content': 'Say exactly: alpha beta gamma delta'}]})
        self.record('anthropic: stop_sequences accepted', status == 200, f'http {status}')

    def probe_misc(self) -> None:
        status, _ = self.request('/health', method='GET')
        self.record('GET /health', status == 200, f'http {status}')
        # Unknown paths under /v1 must be JSON 404s so a client can tell an API miss from a
        # served page. Non-API paths fall through to the bundled WebUI (SPA), which is its own
        # documented behavior and is not probed here.
        status, _ = self.request('/v1/does-not-exist', method='GET')
        self.record('unknown /v1 path returns a 404', status == 404, f'http {status}')
        status, _ = self.request('/v1/chat/completions',
                                 self.chat(messages=[{'role': 'user', 'content': 'x' * 64}]),
                                 headers={'Authorization': 'Bearer local'})
        self.record('Bearer credential accepted when auth is off', status == 200, f'http {status}')

    def run(self) -> int:
        self.probe_models()
        self.probe_cors_preflight()
        self.probe_chat_basic()
        self.probe_chat_json_object()
        self.probe_chat_controls()
        self.probe_chat_tools()
        self.probe_chat_stream()
        self.probe_responses()
        self.probe_anthropic()
        self.probe_misc()
        total = len(self.results)
        print(f'\n{total - self.failures}/{total} probes passed')
        return 1 if self.failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://127.0.0.1:8088')
    parser.add_argument('--model', default='qwen3.8-27b')
    args = parser.parse_args()
    return Probes(args.base, args.model).run()


if __name__ == '__main__':
    sys.exit(main())
