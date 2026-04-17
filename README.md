# LLM Council

![llmcouncil](header.jpg)

This project is inspired by **Andrej Karpathy's** concept of [reading books together with LLMs](https://x.com/karpathy/status/1990577951671509438). Instead of asking a single LLM, you can assemble them into a "LLM Council" that deliberates together.

This is a simple, local web app that essentially looks like ChatGPT, but with a 3-stage deliberation system:
- Multiple LLMs answer independently
- They anonymously review and rank each other's work
- A Chairman LLM synthesizes the best response

The beauty of this approach is that you get diverse perspectives, anonymous peer review (preventing bias), and a final synthesis—all without needing expensive proprietary APIs.

In a bit more detail, here is what happens when you submit a query:

1. **Stage 1: First opinions**. The user query is given to all LLMs individually, and the responses are collected. The individual responses are shown in a "tab view", so that the user can inspect them all one by one.
2. **Stage 2: Review**. Each individual LLM is given the responses of the other LLMs. Under the hood, the LLM identities are anonymized so that the LLM can't play favorites when judging their outputs. The LLM is asked to rank them in accuracy and insight.
3. **Stage 3: Final response**. The designated Chairman of the LLM Council takes all of the model's responses and compiles them into a single final answer that is presented to the user.

## Vibe Code Alert

This project was 99% vibe coded as a fun Saturday hack because I wanted to explore and evaluate a number of LLMs side by side in the process of [reading books together with LLMs](https://x.com/karpathy/status/1990577951671509438). It's nice and useful to see multiple responses side by side, and also the cross-opinions of all LLMs on each other's outputs. I'm not going to support it in any way, it's provided here as is for other people's inspiration and I don't intend to improve it. Code is ephemeral now and libraries are over, ask your LLM to change it in whatever way you like.

## Setup

### 1. Install Dependencies

The project uses [uv](https://docs.astral.sh/uv/) for project management.

**Backend:**
```bash
uv sync
```

**Frontend:**
```bash
cd frontend
npm install
cd ..
```

### 2. Configure API Key

Create a `.env` file in the project root:

```bash
GROQ_API_KEY=gsk_...
```

Get your free API key at [console.groq.com](https://console.groq.com/). Groq offers generous free tier rates for fast inference with no payment required.

### 3. Configure Models (Optional)

Edit `backend/config.py` to customize the council. The current setup uses Groq's production models:

```python
COUNCIL_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
]

CHAIRMAN_MODEL = "openai/gpt-oss-120b"
```

See [Groq's model list](https://console.groq.com/docs/models) for currently available models.

## Running the Application

**Option 1: Use the start script**
```bash
./start.sh
```

**Option 2: Run manually**

Terminal 1 (Backend):
```bash
uv run python -m backend.main
```

Terminal 2 (Frontend):
```bash
cd frontend
npm run dev
```

Then open http://localhost:5173 in your browser.

## Tech Stack

- **Backend:** FastAPI (Python 3.10+), async httpx, Groq API
- **Frontend:** React + Vite, react-markdown for rendering
- **Storage:** JSON files in `data/conversations/`
- **Package Management:** uv for Python, npm for JavaScript
- **LLM Provider:** Groq (free tier with no payment required)
