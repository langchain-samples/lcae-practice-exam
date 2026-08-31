# LCAE Practice Exam Traces

This repo seeds a small set of LangSmith traces into your workspace so you can
answer the trace-investigation questions on the LangChain Certified Agent
Engineer (LCAE) practice exam.

## Setup

1. Clone the repo and enter it:

   ```sh
   git clone https://github.com/langchain-samples/lcae-practice-exam.git
   cd lcae-practice-exam
   ```

2. Copy `.env.example` to `.env` and set your LangSmith API key:

   ```sh
   cp .env.example .env
   # then edit .env and paste in your LANGSMITH_API_KEY
   ```

   If your LangSmith workspace is outside the default US region, also set
   `LANGSMITH_ENDPOINT` in `.env` to your regional API endpoint.

3. Install dependencies and upload the traces:

   ```sh
   uv sync
   uv run python upload_traces.py
   ```

The script uploads every trace in `traces.json` into a LangSmith project named
`customer-support` in your workspace. Open that project in the LangSmith UI
to answer the practice exam questions.
