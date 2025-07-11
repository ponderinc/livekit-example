# Simple Conversational Bot using Ponder with LiveKit

A LiveKit agent that uses Ponder for TTS

Go to https://useponder.ai to get your API KEY

## Environment Variables

Copy `env.example` to `.env` and configure:

```ini
# Required API Keys
LIVEKIT_URL=
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=
OPENAI_API_KEY=
PONDER_API_KEY=
```

## Running the Server

Set up and activate your virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the agent:

```bash
python agent.py console
```

This will start the agent in your terminal that you can talk with.


