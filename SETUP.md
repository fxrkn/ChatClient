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

**Cloud:** add your API key from inside the app (the key is checked before it is saved):

```
/key set openrouter
```

Keys are stored in the OS keychain (Windows Credential Manager, macOS Keychain,
GNOME Keyring/KWallet). Where no keychain is available, for example on a Linux server,
they go to `keys.json` in the config folder, readable only by your user.
Environment variables such as `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`,
`OPENAI_API_KEY` or `GEMINI_API_KEY` also work and take priority.

### Data

Characters, personas and chats are plain files in your user data folder
(Windows: `%LOCALAPPDATA%\ChatClient`, Linux: `~/.local/share/ChatClient`):

```
characters/<name>/card.json, chats/*.jsonl, memory.jsonl
personas/<name>/persona.json
plain-chats/*.jsonl          (chats without a character)
```

`/open` shows a folder, `/char export` and `/import` move characters and chats between
machines. For a portable install (e.g. on a USB stick), create an empty file named
`portable` next to the `.venv` folder: data then goes to `ChatClientData` beside it.

### Development

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
```
