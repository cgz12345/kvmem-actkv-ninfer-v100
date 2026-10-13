"""ActKV A/B harness: one deterministic multi-turn workload, identical prompts.

The haystack, needle placement, questions and plant positions all come from a
seeded PRNG and fixed templates, so both arms receive byte-identical prompts by
construction. The chat is continued by resending the full messages array on
every turn (the protocol the five-case engine test used; prefix + KVMem history
reuse verified there).

Usage:
  python actkv-ab.py plan [dump_tokens]                 # print the plan, no server
  python actkv-ab.py run <arm:on|off> <label> [port] [dump_tokens]

Writes <out>/<label>.json (default actkv-ab/ under the current directory) and
prints a one-line summary. The server may live on any port; the workload and
its prompts do not depend on the server configuration.
"""
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

OUT = r'D:\KVMem-V100-Project\runtime-logs\actkv-ab'
SEED = 20261013
DEFAULT_PORT = 8089
DEFAULT_DUMP_TOKENS = 12000
CHARS_PER_TOKEN = 3.8  # calibrated for the generated code-like text

SYS = ("You are a maintenance assistant for a large service repository. "
       "Always answer with the value only, no explanation or restatement.")


def make_rng(tag):
    return random.Random(f"{SEED}:{tag}")


# (id, keyword, value) — 16 code needles. 12 are asked, 4 are decoys that only
# occupy window pages and compete with the asked ones during selection.
CODE_NEEDLES = [
    ("A1", "ingest-v3 retry budget", "5417"),
    ("A2", "rebalance-lag-s7 threshold", "8412"),
    ("B1", "tenant-shard-map-v7 batch size", "96233"),
    ("B2", "token-cache-ttl", "4489"),
    ("C1", "audit salt prefix", "sk_live_9f42c"),
    ("C2", "backoff profile", "exp-jitter-8"),
    ("D1", "checkpoint interval", "7714"),
    ("D2", "deadline epoch", "2031-04-19"),
    ("E1", "rebalance completion signal", "SIG_REBALANCE_DONE"),
    ("E2", "ingest topic", "ingest.events.v9"),
    ("F1", "eu-west queue", "ingest-v3-eu-west-7"),
    ("F2", "archive bucket", "s3://acid-green-data"),
    ("G1", "cron window", "*/7 3 * * 1-5"),
    ("G2", "shard rebalance feature flag", "FEATURE_SHARD_REBALANCE_V2"),
    ("H1", "rebalance lag metric", "ingest.rebalance.lag_seconds"),
    ("H2", "module owner alias", "alina.kovacs"),
]

# (id, name, value) — conversational facts planted in a turn and asked later.
CONV_FACTS = [
    ("c1", "staging tenant identifier", "t-9f42c"),
    ("c2", "release channel", "canary-eu-3"),
    ("c3", "alert webhook id", "wh-88213"),
    ("c4", "escalation roster", "ops-tier2"),
]

# Turn plan: (dump index, needles in dump, conv fact planted, question ids).
# Each question is a needle/fact never asked before, so no earlier answer in the
# conversation can leak the value into the prompt.
TURN_PLAN = [
    (1, ["A1", "A2"], None, ["A1"]),
    (2, ["B1", "B2"], "c1", ["A2"]),
    (3, ["C1", "C2"], "c2", ["B1"]),
    (4, ["D1", "D2"], "c3", ["c1"]),
    (5, ["E1", "E2"], None, ["B2"]),
    (6, ["F1", "F2"], "c4", ["c2"]),
    (7, ["G1", "G2"], None, ["D1"]),
    (8, ["H1", "H2"], None, ["C1", "C2", "E1", "c3", "c4"]),
]

CODE_BY_ID = {item[0]: item for item in CODE_NEEDLES}
CONV_BY_ID = {item[0]: item for item in CONV_FACTS}


def needle_line(item):
    _, keyword, value = item
    return (f"# resident note [alina.kovacs]: the {keyword} is {value} before "
            f"escalation; do not change without sign-off.")


def question_text(qid):
    if qid in CODE_BY_ID:
        _, keyword, _ = CODE_BY_ID[qid]
        return f"What is the {keyword}?"
    _, name, _ = CONV_BY_ID[qid]
    return f"What is the {name}?"


def expected_text(qid):
    return (CODE_BY_ID.get(qid) or CONV_BY_ID.get(qid))[2]


def gen_dump(index, rng, chars_target):
    """Deterministic code-like module text of roughly chars_target characters."""
    lines = [f"# module: ingest_worker_{index:02d} (revision r{index}{rng.randrange(100, 999)})",
             "from __future__ import annotations", ""]
    size = sum(len(line) + 1 for line in lines)
    n = 0
    while size < chars_target:
        n += 1
        kind = rng.randrange(10)
        if kind < 3:
            line = (f"    def process_batch_{index}_{n}(self, records: list[Record], "
                    f"*, shard: int = {rng.randrange(0, 64)}, retries: int = {rng.randrange(1, 9)}) "
                    f"-> BatchResult:")
        elif kind < 5:
            line = (f"        self.log.info(\"shard=%s batch=%s records=%s\", shard, "
                    f"{rng.randrange(1000, 9999)}, len(records))")
        elif kind < 7:
            line = (f"    BACKLOG_LIMIT_{index}_{n} = {rng.randrange(100, 99999)}  "
                    f"# tuned on the eu-west fleet, see ticket OPS-{rng.randrange(1000, 9999)}")
        elif kind < 9:
            line = (f"        if records and shard % {rng.randrange(2, 9)} == 0: "
                    f"await self.rebalance_pipeline({rng.randrange(1, 64)}, force={rng.choice(['True', 'False'])})")
        else:
            line = (f"    # NOTE: keep the {rng.choice(['hash', 'window', 'cursor', 'lease'])} "
                    f"logic in sync with shard_router_v{rng.randrange(1, 9)}")
        lines.append(line)
        size += len(line) + 1
    lines.append("")
    return "\n".join(lines)


def build_turns(dump_tokens):
    chars_target = int(dump_tokens * CHARS_PER_TOKEN)
    rng = make_rng('dumps')
    turns = []
    for index, needle_ids, conv_id, question_ids in TURN_PLAN:
        dump = gen_dump(index, rng, chars_target)
        lines = dump.split("\n")
        # Plant each needle at 25% / 75% of the module body (before the trailing
        # blank line), so both depths and the middle of the dump carry facts.
        for position, needle_id in zip((0.25, 0.75), needle_ids):
            at = max(3, int(len(lines) * position))
            lines.insert(at, needle_line(CODE_BY_ID[needle_id]))
        dump = "\n".join(lines)
        parts = [f"File drop {index} for review follows.\n\n```python\n{dump}\n```"]
        if conv_id:
            _, name, value = CONV_BY_ID[conv_id]
            parts.append(f"Session note: the {name} for this deployment is {value}.")
        if len(question_ids) == 1:
            parts.append(f"Question: {question_text(question_ids[0])}\nAnswer with the value only.")
        else:
            numbered = "\n".join(f"{i}. {question_text(qid)}"
                                 for i, qid in enumerate(question_ids, 1))
            parts.append(f"Questions:\n{numbered}\nAnswer each with its value only.")
        turns.append({'dump_index': index, 'conv': conv_id, 'questions': question_ids,
                      'content': "\n\n".join(parts)})
    return turns


def normalize(text):
    return ''.join(ch for ch in text.lower() if ch not in ' ,')


def http_json(url, body, timeout):
    data = json.dumps(body, ensure_ascii=False).encode('utf-8')
    request = urllib.request.Request(url, data=data,
                                     headers={'Content-Type': 'application/json; charset=utf-8'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.loads(response.read().decode('utf-8'))


def health(port):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=5) as resp:
            return resp.status
    except Exception as error:  # noqa: BLE001
        return f'DOWN ({type(error).__name__})'


def run_arm(arm, label, port, dump_tokens):
    turns = build_turns(dump_tokens)
    url = f'http://127.0.0.1:{port}/v1/chat/completions'
    messages = [{'role': 'system', 'content': SYS}]
    record = {'label': label, 'arm': arm, 'port': port, 'dump_tokens': dump_tokens,
              'seed': SEED, 'started': time.strftime('%Y-%m-%d %H:%M:%S'),
              'health_before': health(port), 'turns': []}
    if record['health_before'] != 200:
        raise SystemExit(f'server is not healthy: {record["health_before"]}')
    wall_start = time.time()
    for number, turn in enumerate(turns, 1):
        messages.append({'role': 'user', 'content': turn['content']})
        body = {'model': 'qwen3.8-27b', 'messages': messages, 'max_tokens': 96,
                'temperature': 0.0, 'reasoning_effort': 'none'}
        started = time.time()
        try:
            status, response = http_json(url, body, timeout=1800)
        except urllib.error.HTTPError as error:
            detail = error.read().decode('utf-8', 'replace')[:400]
            raise SystemExit(f'turn {number}: HTTP {error.code}: {detail}')
        seconds = round(time.time() - started, 1)
        choice = response['choices'][0]
        answer = choice['message'].get('content') or ''
        usage = response.get('usage') or {}
        timings = response.get('timings') or {}
        messages.append({'role': 'assistant', 'content': answer})
        norm = normalize(answer)
        passes = {qid: normalize(expected_text(qid)) in norm for qid in turn['questions']}
        entry = {'turn': number, 'dump_index': turn['dump_index'], 'questions': turn['questions'],
                 'passes': passes, 'answer': answer, 'finish_reason': choice.get('finish_reason'),
                 'seconds': seconds, 'prompt_tokens': usage.get('prompt_tokens'),
                 'cached_tokens': (usage.get('prompt_tokens_details') or {}).get('cached_tokens'),
                 'completion_tokens': usage.get('completion_tokens'),
                 'prompt_per_second': timings.get('prompt_per_second'),
                 'decode_per_second': timings.get('predicted_per_second'),
                 'prompt_n': timings.get('prompt_n'), 'predicted_n': timings.get('predicted_n')}
        record['turns'].append(entry)
        print(f"turn {number}: {seconds}s prompt={entry['prompt_tokens']} "
              f"cached={entry['cached_tokens']} {entry['passes']} answer={answer[:60]!r}",
              flush=True)
    record['wall_seconds'] = round(time.time() - wall_start, 1)
    record['health_after'] = health(port)
    asked = [qid for turn in record['turns'] for qid in turn['questions']]
    record['questions_total'] = len(asked)
    record['questions_passed'] = sum(1 for turn in record['turns']
                                     for ok in turn['passes'].values() if ok)
    record['failures'] = [qid for turn in record['turns']
                          for qid, ok in turn['passes'].items() if not ok]
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f'{label}.json')
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
    total_seconds = sum(t['seconds'] for t in record['turns'])
    prompt_total = sum(t['prompt_tokens'] or 0 for t in record['turns'])
    decode_speeds = [t['decode_per_second'] for t in record['turns'] if t['decode_per_second']]
    print(f"SUMMARY arm={arm} label={label} passed="
          f"{record['questions_passed']}/{record['questions_total']} "
          f"failures={record['failures']} wall={record['wall_seconds']}s "
          f"turn_sum={total_seconds:.0f}s prompt_total={prompt_total} "
          f"decode_median={sorted(decode_speeds)[len(decode_speeds)//2]:.1f} tok/s "
          f"-> {path}", flush=True)
    return record


def print_plan(dump_tokens):
    turns = build_turns(dump_tokens)
    for number, turn in enumerate(turns, 1):
        print(f"turn {number}: dump {turn['dump_index']} chars={len(turn['content'])} "
              f"questions={turn['questions']} conv={turn['conv']}")
    total_chars = sum(len(turn['content']) for turn in turns)
    print(f"total chars={total_chars} (approx tokens={int(total_chars / CHARS_PER_TOKEN)})")
    for qid in [q for turn in turns for q in turn['questions']]:
        print(f"  {qid}: {question_text(qid)} -> {expected_text(qid)}")


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    mode = sys.argv[1]
    if mode == 'plan':
        print_plan(int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DUMP_TOKENS)
        return 0
    if mode == 'run':
        arm, label = sys.argv[2], sys.argv[3]
        port = int(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_PORT
        dump_tokens = int(sys.argv[5]) if len(sys.argv) > 5 else DEFAULT_DUMP_TOKENS
        run_arm(arm, label, port, dump_tokens)
        return 0
    raise SystemExit(__doc__)


if __name__ == '__main__':
    sys.exit(main())
