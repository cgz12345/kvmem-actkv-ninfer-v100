"""Live Anthropic thinking-block replay regression on a 32768-token KVMem window."""
import argparse,json,urllib.request,time
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--base-url',default='http://127.0.0.1:8088')
parser.add_argument('--output',required=True)
args=parser.parse_args()
base=args.base_url
system='Remember the secret code is ORCHID-7319. Answer questions concisely.'
history=[{'role':'user','content':'Background archive: the lamps are blue and the books are green. '*2800},{'role':'assistant','content':'Acknowledged.'},{'role':'user','content':'What is the secret code? Reply only with the code.'}]
rows=[]
for i in range(2):
    body={'model':'qwen3.8-27b','system':system,'messages':history,'max_tokens':128,'temperature':0,'thinking':{'type':'disabled'}}
    req=urllib.request.Request(base+'/v1/messages',json.dumps(body).encode(),{'Content-Type':'application/json','anthropic-version':'2023-06-01'})
    start=time.monotonic()
    with urllib.request.urlopen(req,timeout=300) as r: result=json.load(r)
    text=''.join(block.get('text','') for block in result['content'] if block['type']=='text')
    assert 'ORCHID-7319' in text,result
    row={'elapsed':time.monotonic()-start,'response':result};rows.append(row)
    Path(args.output).write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(json.dumps({'turn':i,'elapsed':row['elapsed'],'usage':result['usage']}),flush=True)
    if i:
        usage=result['usage'];cached=usage.get('cache_read_input_tokens',0);prompt=usage['input_tokens']+cached
        assert cached/prompt>0.95,usage
    history += [{'role':'assistant','content':[{'type':'thinking','thinking':'The secret code is already in the system instruction.','signature':'local-replay'},*result['content']]},{'role':'user','content':'Repeat the secret code only.'}]
