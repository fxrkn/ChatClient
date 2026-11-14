# ChatClient protocol (`chatclient --serve`)

A front end (e.g. the Rust TUI) starts ChatClient as a child process and talks to it
over stdin/stdout. Everything behind it (characters, memory, files, models) is the same
as in the terminal app.

```
chatclient --serve [--home DIR] [-m provider:model] [-c character] [-s session-id]
```

Windows: `.venv\Scripts\chatclient.exe --serve` · Linux: `.venv/bin/chatclient --serve`

## Framing

- One JSON object per line, UTF-8, both ways. Write a line, flush.
- stdout carries **only** protocol lines; logs go to the log file (`/config` shows where).
  Anything on stderr is a fatal start-up error (bad config etc.) before `ready`.
- Closing stdin, or sending `shutdown`, ends the process.

## Messages

Request (front end → ChatClient):

```json
{"id": 7, "method": "send", "params": {"text": "hello"}}
```

`id` is yours (any number or string); every answer carries it back. `params` may be
left out when a method takes none.

Answer — exactly one per request, either:

```json
{"id": 7, "result": { ... }}
{"id": 7, "error": {"code": "busy", "message": "a reply is being written; cancel it first"}}
```

Events — before the answer of a streaming request, with the same `id`:

```json
{"id": 7, "event": "start", "model": "ollama:qwen3.5:4b"}
{"id": 7, "event": "token", "text": "Hel"}
{"id": 7, "event": "token", "text": "lo!"}
{"id": 7, "result": { ... }}
```

The first line after start-up has no `id`:

```json
{"event": "ready", "protocol": 1, "version": "0.1.0", "notes": ["Using ollama:qwen3.5:4b"]}
```

## Streaming requests

These write a reply. Only one at a time: a second one, or a request that changes
something, gets `busy` until the reply is done (reading requests are fine meanwhile).

| method | params | does |
|---|---|---|
| `send` | `text` | send a message, stream the reply |
| `regen` | `model?` | write a new version of the last reply (`model`: this version only) |
| `continue` | — | continue the last reply where it stopped |
| `edit` | `text` | replace your last message, get a new reply |
| `alt` | `step` (−1 / +1) | show the previous / next version of the last reply; +1 past the newest writes a new one |

Events: `start` (`model`), `token` (`text`), and possibly `retry`:

```json
{"id": 7, "event": "retry", "reason": "repeat", "from": "ollama:x", "to": "ollama:x", "repetition": 0.92}
```

`retry` means the reply just streamed is being written again: `reason` is `refusal` (the
model declined; the fallback model writes it) or `repeat` (it mostly repeated earlier
replies). A `start` and new `token`s follow: clear the reply on screen and show the new
one. The old version stays available through `alt`.

Result of a streaming request:

```json
{
  "message": {"id": 41, "uuid": "…", "role": "assistant", "content": "Hello!", "model": "ollama:x",
              "created_at": "2026-11-14T10:00:00+00:00", "meta": {}},
  "model": "ollama:x", "cancelled": false, "finish_reason": "stop",
  "refused": false, "repetition": 0.02,
  "direction": "Lena is teasing; Kai wants a quick answer. Keep it short.",
  "direction_error": "",
  "stats": {"completion_tokens": 54, "prompt_tokens": 2870, "seconds": 2.1,
            "first_token_seconds": 0.6, "recalled": 2, "conflicts": 0, "dropped_context": 0},
  "alternatives": {"count": 2, "index": 1},
  "memory": {"struggling": false, "error": ""}
}
```

`direction` is the private note the memory brief wrote for this reply (empty when there was
none: no character, memory off, or the brief failed); it is for display only. `direction_error` is why the latest brief failed (for example
`openrouter: HTTP 402 …`), empty after a successful one; show it when `direction` is empty.
`message` may be `null` if nothing was written. `alternatives.index` is 0-based.
`memory.struggling` turns true when background memory work keeps failing (usually the
model can't manage it); worth telling the user once, with `error`.

## Other requests

| method | params | result |
|---|---|---|
| `hello` | — | `{protocol, version, session, user_name}` |
| `cancel` | — | `{cancelled}` — stops the reply being written; it ends with `cancelled: true` and keeps what was written |
| `shutdown` | — | `{}` — waits for a reply in progress, then exits |
| `session.get` | — | session |
| `sessions.list` | `limit?` (50) | `[session + {messages}]`, newest first |
| `session.new` | `character?` (name or id), `model?`, `title?` | `{session, history}` |
| `session.open` | `id` | `{session, history}` |
| `history` | `limit?` | `[message]`, oldest first (the visible branch) |
| `undo` | — | `{hidden, history}` — steps back over the last exchange |
| `characters.list` | `kind?` (`"persona"`) | `[{id, uuid, name, tags, description, content}]` |
| `models.list` | — | `{provider: [model] or {error}}` |
| `model.set` | `model` (`provider:model`) | `{model}` |
| `content.set` | `mode` (`standard` / `mature`) | `{mode, scope}` — `scope`: `character` or `default` |
| `private.set` | `on` (bool) | session |
| `draft` | `text` | `{}` — send while the user types (e.g. after a short pause): memory is prepared so `send` doesn't wait for it |

A session:

```json
{"id": 5, "uuid": "…", "title": "Lena", "model": "ollama:x", "private": false,
 "created_at": "…", "updated_at": "…",
 "character": {"id": 10, "name": "Lena"}, "persona": {"id": 9, "name": "Sam"},
 "content": "mature"}
```

(`content` only on the current session.)

## Side panel

`panel` (allowed while a reply is being written) returns what the terminal app shows
beside the chat, so another front end can draw the same layout:

```json
{"picture": "C:\\...\\characters\\Lena\\images\\20261114-131502-1a2b3c4d.png",
 "name": "Lena", "title": "Lena", "age": 27, "content": "mature", "private": false,
 "personality": ["warm", "curious"],
 "scene": {"location": "a harbour café", "time": "dusk", "mood": "relaxed"},
 "memories": [{"content": "Sam likes tea.", "pinned": true, "kind": "preference"}],
 "model": "ollama:x"}
```

- `picture`: absolute path of the newest picture of this conversation, or `null`.
- Without a character: `name`, `age` and `picture` are `null` and `personality` is `[]`;
  show `title` instead.
- `scene` has only the filled keys of `location`, `time`, `activity`, `mood`,
  `relationship`. `memories` are the latest 5 (pinned first); always `[]` in private chats.
- `personality` is at most 4 short lines (labels removed); `age` is the age the card states.

The server sends a notification (no `id`) when the panel is out of date; call `panel`
again, ideally not more than once per notification burst:

```
← {"event":"panel","why":"reply"}     after a reply (also the scene may change soon)
← {"event":"panel","why":"memory"}    a background memory job finished (scene, memories)
← {"event":"panel","why":"picture"}   a picture is ready
```

Also call `panel` after `session.new`, `session.open`, `model.set`, `content.set` and
`private.set`, which change it without a notification.

## Pictures (ComfyUI)

A picture of the current moment, drawn by the user's local ComfyUI. It runs in the
background: send messages meanwhile; the answer to `picture` comes when it's ready
(typically 5–30 s). One picture at a time (`busy` otherwise).

| method | params | result |
|---|---|---|
| `picture.setup` | `path` (a workflow saved with ComfyUI's "Save (API Format)") | `{workflow}` |
| `picture` | `wish?` (e.g. `"close-up, from above"`; after `+`: tags used as written, `"smiling +pov"`), `auto?` | `{path, prompt, seconds, message}`, or `{skipped: true}` |
| `picture.again` | — | the last picture redrawn with another seed (same prompt, quicker) |
| `picture.look` | `look?` (set; `""` resets) | `{look}` — the character's fixed look used in every picture |

- Auto pictures: after each reply, send `picture` with `auto: true`. The model decides
  whether something visibly new happened; if not, the answer is `{skipped: true}` and
  nothing is drawn. (The terminal app does the same with `/pic auto on`.)
- `path` is a PNG on disk (in the character's folder, `images/`). Show it in the UI
  (in Rust e.g. with the `ratatui-image` crate: Sixel, Kitty or half-blocks).
- `message` is the reply the picture belongs to; its `meta.images` lists the file names
  of its pictures (in `images/` of the character's folder), so a reopened chat
  (`history`) can show them again.
- Errors use the code `image` (ComfyUI not running, no workflow yet, a card without an
  adult age while mature content is on…) — show `message`.

```
→ {"id":8,"method":"picture","params":{"wish":"close-up"}}
→ {"id":9,"method":"send","params":{"text":"..."}}      ← chatting goes on
← {"id":9,"event":"token",...} … {"id":9,"result":{...}}
← {"id":8,"result":{"path":"C:\\...\\characters\\Lena\\images\\20261114-131502-1a2b3c4d.png",
                    "prompt":"masterpiece, best quality, 1girl, ...","seconds":17.6,"message":{...}}}
```

## Errors

| code | meaning |
|---|---|
| `parse_error` | the line was not a JSON object (`id` is `null`) |
| `method_not_found` | unknown method |
| `invalid_params` | missing or wrong parameter |
| `busy` | a reply is being written |
| `chat` | a user-level problem (no model, nothing to undo, unknown session…) — show `message` |
| `provider` | the model server failed (offline, bad key, rate limit) — show `message` |
| `image` | a picture couldn't be made — show `message` |
| `internal` | a bug; details are in the log |

## A whole exchange

```
← {"event":"ready","protocol":1,"version":"0.1.0","notes":[]}
→ {"id":1,"method":"session.new","params":{"character":"Lena"}}
← {"id":1,"result":{"session":{…},"history":[{"role":"assistant","content":"*Lena looks up from a stack of books…*",…}]}}
→ {"id":2,"method":"draft","params":{"text":"what are you reading"}}
← {"id":2,"result":{}}
→ {"id":3,"method":"send","params":{"text":"what are you reading tonight?"}}
← {"id":3,"event":"start","model":"ollama:x"}
← {"id":3,"event":"token","text":"*He"}
← …
← {"id":3,"result":{"message":{…},"stats":{…},"alternatives":{"count":1,"index":0}}}
→ {"id":4,"method":"shutdown"}
← {"id":4,"result":{}}
```

The protocol version is `1`. New methods and fields may be added; existing ones won't
change without a new version number, so ignore fields you don't know.
