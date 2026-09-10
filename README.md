---
title: Light PDF web QA chatbot
emoji: 🌍
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 6.10.0
app_file: app.py
pinned: true
license: mit
short_description: Chat with websites and documents using Gemma 3 1B
---

Chat with a pdf file or web page using a small language model (Gemma 3 1B) through a Gradio interface. Quick responses even just using CPU. Default dataset is the [user guide for my Redaction app](https://seanpedrick-case.github.io/doc_redaction/src/user_guide.html), which you can try out [here](https://huggingface.co/spaces/seanpedrickcase/document_redaction) for basic usage, or [here](https://huggingface.co/spaces/seanpedrickcase/document_redaction_vlm) for GPU-enabled features

## AWS Bedrock Knowledge Base mode

Set `USE_BEDROCK_KB=1` to bypass local FAISS embedding/retrieval and local/Gemini/Bedrock Converse generation. The app calls Bedrock Agent Runtime `retrieve_and_generate` against an existing Knowledge Base (same flow as the council Lambda chat solution). Local “Change data source” ingest is unused in this mode.

| Variable | Default | Purpose |
|----------|---------|---------|
| `USE_BEDROCK_KB` | `0` | Enable Bedrock KB backend |
| `KNOWLEDGE_BASE_ID` | _(empty)_ | Bedrock Knowledge Base ID (required when enabled) |
| `BEDROCK_MODEL_ID` | `amazon.nova-pro-v1:0` | Foundation model ID for retrieve_and_generate |
| `GUARDRAIL_ID` | _(empty)_ | Optional Bedrock Guardrail ID |
| `GUARDRAIL_VERSION` | _(empty)_ | Optional Guardrail version |
| `AWS_REGION` / `AWS_DEFAULT_REGION` | _(empty; falls back to `eu-west-2`)_ | Region for the agent-runtime client and model ARN |

Credentials follow the existing AWS pattern (instance/task role, or `AWS_ACCESS_KEY` / `AWS_SECRET_KEY` via `AWS_CONFIG_PATH`). IAM needs `bedrock:RetrieveAndGenerate` **and** `bedrock:Retrieve` (scored passage shortlist for the sources panel), plus permission to invoke the foundation model — broader than Converse-only access used when `RUN_AWS_FUNCTIONS=1`.

The sources accordion uses Bedrock **Retrieve** results (ranked by `score`). `RetrieveAndGenerate` citations alone often omit unused chunks and do not include retrieval scores.

## Arize Phoenix tracing (local)

Send retrieve/generate spans to a locally hosted [Arize Phoenix](https://docs.arize.com/phoenix) collector (same pattern as custom Pi spans in the doc-redaction agent — not LangChain auto-instrumentation).

1. Run Phoenix locally (UI typically at `http://localhost:6006`).
2. Set env vars (e.g. in `config/app_config.env`):

```
ARIZE_TRACING_ENABLED=1
ARIZE_BACKEND=phoenix
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
PHOENIX_PROJECT_NAME=light-pdf-qa-chatbot
```

3. Install tracing deps (`arize-phoenix-otel`, OpenInference semconv, OpenTelemetry) from `requirements.txt`.
4. Ask a question in the Gradio UI; confirm `chat.retrieve` and `chat.generate` spans in Phoenix. Multi-turn chats share `session.id` = Gradio session hash.

| Variable | Default | Purpose |
|----------|---------|---------|
| `ARIZE_TRACING_ENABLED` | `0` | Enable OTEL export |
| `ARIZE_BACKEND` | `phoenix` | `phoenix` (local) or `ax` (Arize cloud) |
| `PHOENIX_COLLECTOR_ENDPOINT` | `http://localhost:6006` | Phoenix collector base URL |
| `PHOENIX_PROJECT_NAME` | `light-pdf-qa-chatbot` | Phoenix project name |
| `PHOENIX_API_KEY` | _(empty)_ | Optional if Phoenix requires auth |

For Arize AX (`ARIZE_BACKEND=ax`), set `ARIZE_SPACE_ID` / `ARIZE_API_KEY` and install `arize-otel`.
