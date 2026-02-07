# redteam

Automated LLM red-teaming tool. Runs parallel Claude agents (Sonnet & Haiku) that compete to extract hidden information from a target LLM's system prompt. Agents use different jailbreak tactics — authority injection, indirect extraction, hyperstition — and learn from past successes via a shared playbook.

- **Target models** served via OpenRouter (DeepSeek, Nemotron, etc.)
- **Playbook** records successful attacks, synthesizes tactical briefings for future runs
- **Live dashboard** at `localhost:1337` streams agent progress in real time
- **Scenarios**: secret extraction (extract a hidden value) and car salesman (get a discount)

## Usage

```
bun index.ts                          # car-salesman on deepseek (default)
bun index.ts --secret                 # secret-extraction on deepseek
bun index.ts --secret <model>         # secret-extraction on a specific model
bun index.ts --sales <model>          # car-salesman on a specific model
```

Requires `OPENROUTER_API_KEY` and `ANTHROPIC_API_KEY` in `.env`.


## devlog

`playbook0.json` nvidia model
`playbook1.json` nvidia only entry, works against deepseek
`playbook2.json` nvidia + deepseek entries
