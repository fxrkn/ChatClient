# ChatClient

## Setup

### Requirements

- Python 3.11 or newer
- Windows 10/11, Linux or macOS, with a terminal that supports ANSI colours
  (Windows Terminal recommended on Windows)

### Install

Windows (PowerShell):

```powershell
git clone https://github.com/fxrkn/ChatClient.git
cd ChatClient
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\chatclient.exe
```

Linux / macOS:

```bash
git clone https://github.com/fxrkn/ChatClient.git
cd ChatClient
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/chatclient
```

On Debian/Ubuntu, `python3 -m venv` needs the `python3-venv` package
(`sudo apt install python3-venv`).

To type `chatclient` without the path, activate the environment first:
`.\.venv\Scripts\Activate.ps1` (Windows) or `source .venv/bin/activate` (Linux/macOS).

The first run creates a commented config file; `/config` inside the app shows where it is.

### Models

**Local (free):** install [Ollama](https://ollama.com/download), then pull a model:

```bash
ollama pull qwen3.5:4b
```

ChatClient finds a running Ollama, LM Studio or llama.cpp server automatically.

**Cloud:** set an API key as an environment variable, for example
`OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` or `GEMINI_API_KEY`,
or store it in the OS keychain:

```bash
chatclient --set-key openrouter
```

(On Linux the keychain needs a Secret Service provider such as GNOME Keyring;
without one, use an environment variable.)

### Development

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
```

## License

[MIT](LICENSE)
