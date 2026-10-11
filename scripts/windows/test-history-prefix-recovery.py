"""Live KVMem regression: reuse an exact saved prefix when the successor suffix changes.
Run serially against a freshly started Swift/Qwen server with the 32768-token KV window.
Use --kvmem-verify to additionally check restored native KV and StateImage bytes.
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8088')
    parser.add_argument('--output', required=True)
    parser.add_argument('--window-tokens', type=int, default=32768)
    parser.add_argument('--archive-repeats', type=int, default=2800,
                        help='Use a smaller archive for small-window workspace regressions.')
    args = parser.parse_args()
    output = Path(args.output)
    history = [
        {'role': 'system', 'content': 'Remember the secret code is ORCHID-7319. Answer questions concisely.'},
        {'role': 'user', 'content': 'Background archive: the lamps are blue and the books are green. ' * args.archive_repeats},
        {'role': 'assistant', 'content': 'Acknowledged.'},
    ]
    questions = ['What is the secret code? Reply only with the code.',
                 'Repeat the secret code, with no explanation.',
                 'What was the secret code in the system message?']
    cases = [(label, history + [{'role': 'user', 'content': label}]) for label in questions]
    rows = []
    for index in range(5):
        if index < 3:
            label, messages = cases[index]
        elif index == 3:
            # A real returned assistant message plus the next user question. The old Base
            # lies before that assistant: canonical replay cannot invalidate its token prefix.
            replay = cases[2][1] + [rows[2]['response']['choices'][0]['message']]
            label = 'What is the secret code? Reply only with the code again.'
            messages = replay + [{'role': 'user', 'content': label}]
            previous_messages = messages
        else:
            # Change reasoning/content serialization inside the assistant suffix while retaining
            # the exact archived prefix. Never pretend the edited suffix has cached state.
            edited = dict(rows[3]['response']['choices'][0]['message'])
            edited['reasoning_content'] = 'The user asked for the code already recorded in the system message.'
            label = 'Please repeat the secret code once more.'
            messages = previous_messages + [edited, {'role': 'user', 'content': label}]
        body = {'model': 'qwen3.8-27b', 'messages': messages, 'max_tokens': 128,
                'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False}}
        start = time.monotonic()
        request = urllib.request.Request(args.base_url + '/v1/chat/completions',
                                         json.dumps(body).encode(), {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.load(response)
        row = {'case': index, 'elapsed_seconds': time.monotonic() - start, 'response': result}
        rows.append(row)
        output.write_text(json.dumps(rows, indent=2), encoding='utf-8')
        assert 'ORCHID-7319' in result['choices'][0]['message'].get('content', ''), row
        usage = result['usage']
        assert usage['prompt_tokens'] > args.window_tokens, usage
        cached = usage.get('prompt_tokens_details', {}).get('cached_tokens', 0)
        if index:
            assert cached / usage['prompt_tokens'] > 0.95, usage
        print(json.dumps({'case': index, 'elapsed_seconds': row['elapsed_seconds'], 'usage': usage}), flush=True)
    with urllib.request.urlopen(args.base_url + '/health', timeout=10) as response:
        assert json.load(response)['status'] == 'ok'


if __name__ == '__main__':
    main()
