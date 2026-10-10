param([string]$BaseUrl='http://127.0.0.1:8088', [string]$Run='baseline')
$ErrorActionPreference='Stop'
$results=@()
foreach ($trial in 1..2) {
    # Distinct leading text prevents Host-history hits between measurement trials.
    $prompt=('Trial '+$trial+'. ')+('The quick brown fox jumps over the lazy dog. '*400)+' Write a numbered list of 200 common English nouns.'
    $body=@{input=$prompt;max_output_tokens=256;temperature=0;reasoning=@{effort='none'}} | ConvertTo-Json -Depth 5
    $timer=[Diagnostics.Stopwatch]::StartNew()
    $response=Invoke-RestMethod -Uri "$BaseUrl/v1/responses" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 180
    $timer.Stop()
    if ($response.usage.input_tokens_details.cached_tokens -ne 0) {throw 'Benchmark reused cached input'}
    $results+=@{run=$Run;trial=$trial;elapsed_seconds=$timer.Elapsed.TotalSeconds;usage=$response.usage}
}
$results | ConvertTo-Json -Depth 8
