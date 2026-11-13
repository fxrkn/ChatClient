"""Interface strings (Turkish / English). The AI itself always answers in the user's language."""

from __future__ import annotations

import locale
import os

STRINGS: dict[str, dict[str, str]] = {
    "welcome": {
        "en": "Type a message · /help commands · /char characters · Ctrl+D quit",
        "tr": "Mesajınızı yazın · /help komutlar · /char karakterler · Ctrl+D çıkış",
    },
    "theme_hint": {
        "en": "/theme <name> to switch · /theme reset · fine-tune colours in config [ui.colors]",
        "tr": "Değiştirmek için /theme <ad> · /theme reset · renk ince ayarı: config [ui.colors]",
    },
    "theme_set": {"en": "Theme: {name}", "tr": "Tema: {name}"},
    "theme_unknown": {
        "en": "Unknown theme '{name}' (available: {names})",
        "tr": "Bilinmeyen tema '{name}' (mevcut: {names})",
    },
    "config_created": {
        "en": "Created default config: {path}",
        "tr": "Varsayılan ayar dosyası oluşturuldu: {path}",
    },
    "session_opened": {
        "en": "Session #{id}: {title} · {count} messages",
        "tr": "Oturum #{id}: {title} · {count} mesaj",
    },
    "untitled": {"en": "(untitled)", "tr": "(başlıksız)"},
    "no_model": {
        "en": "No model selected. Use /models to list available models, then /model provider:model.",
        "tr": "Model seçilmedi. Modelleri görmek için /models, seçmek için /model sağlayıcı:model.",
    },
    "autodetected": {
        "en": "Found a local model: {model}",
        "tr": "Yerel model bulundu: {model}",
    },
    "no_local": {
        "en": "No local server found (Ollama / LM Studio / llama.cpp). Try 'echo:echo' to test offline.",
        "tr": "Yerel sunucu bulunamadı (Ollama / LM Studio / llama.cpp). Çevrimdışı denemek için 'echo:echo'.",
    },
    "model_set": {"en": "Model: {model}", "tr": "Model: {model}"},
    "cancelled": {"en": "[stopped]", "tr": "[durduruldu]"},
    "ctrl_c_hint": {
        "en": "(Ctrl+D or /quit to exit)",
        "tr": "(çıkmak için Ctrl+D veya /quit)",
    },
    "unknown_command": {
        "en": "Unknown command: {cmd} — see /help",
        "tr": "Bilinmeyen komut: {cmd} — /help'e bakın",
    },
    "new_session": {"en": "New session #{id}", "tr": "Yeni oturum #{id}"},
    "renamed": {"en": "Renamed to: {title}", "tr": "Yeni ad: {title}"},
    "deleted": {"en": "Deleted session #{id}", "tr": "Oturum #{id} silindi"},
    "confirm_delete": {
        "en": 'Delete session #{id} "{title}" permanently? [y/N] ',
        "tr": 'Oturum #{id} "{title}" kalıcı olarak silinsin mi? [e/H] ',
    },
    "aborted": {"en": "Aborted.", "tr": "Vazgeçildi."},
    "undone": {
        "en": "Removed {n} message(s) from the conversation.",
        "tr": "{n} mesaj geri alındı.",
    },
    "no_results": {"en": "No matches.", "tr": "Sonuç yok."},
    "search_short": {
        "en": "Search needs at least 3 characters.",
        "tr": "Arama için en az 3 karakter gerekli.",
    },
    "system_reset": {
        "en": "System prompt reset to default.",
        "tr": "Sistem istemi varsayılana döndü.",
    },
    "system_set": {"en": "System prompt updated.", "tr": "Sistem istemi güncellendi."},
    "param_set": {"en": "{name} = {value}", "tr": "{name} = {value}"},
    "param_reset": {"en": "{name} reset to default", "tr": "{name} varsayılana döndü"},
    "listing_models": {"en": "Querying providers…", "tr": "Sağlayıcılar sorgulanıyor…"},
    "dropped_context": {
        "en": "{n} older message(s) did not fit in the context window",
        "tr": "{n} eski mesaj bağlam penceresine sığmadı",
    },
    "refusal": {
        "en": "[the model declined to answer — try /regen or another model]",
        "tr": "[model yanıt vermeyi reddetti — /regen ya da başka bir model deneyin]",
    },
    "truncated_len": {
        "en": "[reply hit max_tokens — raise it with /set max_tokens N]",
        "tr": "[yanıt max_tokens sınırına ulaştı — /set max_tokens N ile artırın]",
    },
    "you": {"en": "You", "tr": "Sen"},
    "model_load": {"en": "model load {s}s", "tr": "model yükleme {s}s"},
    "builtin": {"en": "built-in", "tr": "hazır"},
    "char_none": {
        "en": "No character in this session (plain chat). /char list shows characters.",
        "tr": "Bu oturumda karakter yok (düz sohbet). Karakterler için /char list.",
    },
    "char_current": {"en": "Character: {name} (#{id})", "tr": "Karakter: {name} (#{id})"},
    "char_list_hint": {
        "en": "/char use <name> to start · /char new to create one",
        "tr": "Başlamak için /char use <ad> · yeni karakter için /char new",
    },
    "char_need_ref": {
        "en": "Which character? Give a name or #id.",
        "tr": "Hangi karakter? Ad ya da #numara yazın.",
    },
    "char_not_found": {
        "en": "No character matches '{ref}' (see /char list).",
        "tr": "'{ref}' ile eşleşen karakter yok (/char list).",
    },
    "char_started": {
        "en": "New session #{id} with {name}.",
        "tr": "{name} ile yeni oturum #{id}.",
    },
    "char_created": {"en": "Created {name} (#{id}).", "tr": "{name} oluşturuldu (#{id})."},
    "char_saved": {"en": "Saved {name}.", "tr": "{name} kaydedildi."},
    "char_deleted": {"en": "Deleted {name}.", "tr": "{name} silindi."},
    "confirm_delete_char": {
        "en": "Delete {name}? Its {n} session(s) stay but lose the character. [y/N]",
        "tr": "{name} silinsin mi? {n} oturumu kalır ama karaktersiz olur. [e/H]",
    },
    "edit_hint": {
        "en": "Edit each field (Enter keeps it, Alt+Enter = new line, Ctrl+C cancels).",
        "tr": "Her alanı düzenleyin (Enter korur, Alt+Enter yeni satır, Ctrl+C iptal).",
    },
    "q_name": {
        "en": "Name of the character? (Ctrl+C cancels)",
        "tr": "Karakterin adı ne? (iptal için Ctrl+C)",
    },
    "q_concept": {
        "en": "Who is {name}, in one line? (age, job, situation, e.g. '34, runs a tiny bookshop')",
        "tr": "{name} kim, tek satırda? (yaş, meslek, durum; örn. '34 yaşında, küçük bir sahafı var')",
    },
    "q_temperament": {"en": "{name}'s temperament?", "tr": "{name}'in mizacı?"},
    "q_speech": {"en": "How does {name} talk?", "tr": "{name} nasıl konuşur?"},
    "q_values": {"en": "What does {name} value most?", "tr": "{name} en çok neye değer verir?"},
    "q_fear": {"en": "{name}'s deepest fear?", "tr": "{name}'in en büyük korkusu?"},
    "q_relationship": {
        "en": "How do you two know each other?",
        "tr": "Birbirinizi nereden tanıyorsunuz?",
    },
    "q_genre": {"en": "What kind of story?", "tr": "Nasıl bir hikâye?"},
    "q_place": {
        "en": "Where does the first scene happen? (optional, e.g. 'a rainy harbour café')",
        "tr": "İlk sahne nerede geçiyor? (isteğe bağlı; örn. 'yağmurlu bir liman kahvesi')",
    },
    "choice_hint": {
        "en": "number, or your own words (Enter skips)",
        "tr": "numara ya da kendi kelimeleriniz (Enter atlar)",
    },
    "choice_hint_multi": {
        "en": "up to {n} numbers like 1,3, or your own words",
        "tr": "en fazla {n} numara (örn. 1,3) ya da kendi kelimeleriniz",
    },
    "q_ai_help": {
        "en": "Let the AI write the opening scene and example lines? (2 short requests with your current model) [Y/n]",
        "tr": "Açılış sahnesini ve örnek replikleri yapay zekâ yazsın mı? (mevcut modelle 2 kısa istek) [E/h]",
    },
    "ai_writing": {"en": "Writing {field}…", "tr": "{field} yazılıyor…"},
    "field_first_mes": {"en": "the opening scene", "tr": "açılış sahnesi"},
    "field_mes_example": {"en": "example lines", "tr": "örnek replikler"},
    "field_description": {"en": "the description", "tr": "tanım"},
    "ai_accept_q": {
        "en": "Use this? [Y]es / [n]o / [r]etry",
        "tr": "Bu kullanılsın mı? [E]vet / [h]ayır / [t]ekrar",
    },
    "ai_retry_q": {
        "en": "Nothing came back. Try again? [y/N]",
        "tr": "Yanıt gelmedi. Tekrar denensin mi? [e/H]",
    },
    "file_not_found": {"en": "File not found: {path}", "tr": "Dosya bulunamadı: {path}"},
    "import_failed": {"en": "Could not import: {error}", "tr": "İçe aktarılamadı: {error}"},
    "char_imported": {
        "en": "Imported {name} (#{id}). /char use {name} to start.",
        "tr": "{name} içe aktarıldı (#{id}). Başlamak için /char use {name}.",
    },
    "char_exported": {"en": "Exported {name} → {path}", "tr": "{name} dışa aktarıldı → {path}"},
    "q_first_mes": {
        "en": "{name}'s opening line (Enter to skip):",
        "tr": "{name}'in ilk sözü (atlamak için Enter):",
    },
    "q_example": {
        "en": "One typical line in {name}'s voice (optional, helps a lot):",
        "tr": "{name}'in ağzından tipik bir replik (isteğe bağlı, çok faydalı):",
    },
    "q_start_now": {
        "en": "Start chatting with {name} now? [Y/n]",
        "tr": "{name} ile şimdi sohbete başlansın mı? [E/h]",
    },
    "persona_none": {
        "en": "No persona set. /persona set tells characters who you are.",
        "tr": "Persona tanımlı değil. /persona set ile karakterlere kim olduğunuzu anlatın.",
    },
    "persona_saved": {
        "en": "Persona saved: {name} (used in new sessions too).",
        "tr": "Persona kaydedildi: {name} (yeni oturumlarda da kullanılır).",
    },
    "persona_cleared": {"en": "Persona cleared.", "tr": "Persona kaldırıldı."},
    "q_persona_name": {
        "en": "What should characters call you?",
        "tr": "Karakterler size ne diye hitap etsin?",
    },
    "q_persona_about": {
        "en": "Tell characters about yourself (appearance, age, anything they should know):",
        "tr": "Kendinizden bahsedin (görünüş, yaş, karakterlerin bilmesi gerekenler):",
    },
    "free_no_openrouter": {
        "en": "No OpenRouter provider configured (base_url must contain openrouter.ai).",
        "tr": "OpenRouter sağlayıcısı tanımlı değil (base_url openrouter.ai içermeli).",
    },
    "free_testing": {
        "en": "Testing up to {n} of {total} free models (each test uses 1 free request)…",
        "tr": "{total} ücretsiz modelden en fazla {n} tanesi deneniyor (her deneme 1 ücretsiz istek harcar)…",
    },
    "free_probing": {
        "en": "Looking for a working free model…",
        "tr": "Çalışan bir ücretsiz model aranıyor…",
    },
    "free_selected": {
        "en": "Using {model} ({n} working model(s) saved; /free list shows them).",
        "tr": "{model} seçildi ({n} çalışan model kaydedildi; /free list ile görebilirsiniz).",
    },
    "free_switch": {
        "en": "⟳ {old} is busy → switching to {new}",
        "tr": "⟳ {old} meşgul → {new} modeline geçiliyor",
    },
    "free_none": {
        "en": "No working free model found right now. Try again later or use /free all.",
        "tr": "Şu an çalışan ücretsiz model bulunamadı. Biraz sonra tekrar deneyin ya da /free all.",
    },
    "free_none_cached": {
        "en": "No tested free models yet — run /free.",
        "tr": "Henüz test edilmiş ücretsiz model yok — /free çalıştırın.",
    },
    "help": {
        "en": """Commands
  /new [title]           start a new session
  /sessions              list sessions
  /switch <id>           switch to a session
  /rename <title>        rename the current session
  /delete <id>           delete a session
  /history [n]           show the last n messages (default 20)
  /model [prov:model]    show or change the model for this session
  /models [provider]     list available models
  /providers             list configured providers
  /system [text|reset]   show or set this session's system prompt
  /set [name value]      show or set generation parameters (value 'reset' clears)
  /regen                 generate a new reply to your last message
  /undo                  remove the last exchange
  /search [--all] text   search messages (this session, or all with --all)
  /char list | new | use <name> | show | edit | delete <name>
                         characters: list, create (wizard), start a chat, view, edit, delete
  /char ai [first_mes|mes_example|description]
                         let the AI rewrite one field of the current character
  /char import <file>    import a community card (.json or .png)
  /char export [name]    save a character as a .json card
  /persona [set|clear]   tell characters who you are
  /free [list|all|N]     test free OpenRouter models and switch to the fastest
  /theme [name|reset]    show or switch the colour theme
  /clear                 clear the screen (history is kept)
  /config                show config and data file locations
  /quit                  exit
Keys: Enter send · Alt+Enter new line · Ctrl+C stop a reply · Ctrl+D quit""",
        "tr": """Komutlar
  /new [başlık]          yeni oturum başlat
  /sessions              oturumları listele
  /switch <id>           oturuma geç
  /rename <başlık>       mevcut oturumu yeniden adlandır
  /delete <id>           oturumu sil
  /history [n]           son n mesajı göster (varsayılan 20)
  /model [sağl:model]    bu oturumun modelini göster/değiştir
  /models [sağlayıcı]    kullanılabilir modelleri listele
  /providers             tanımlı sağlayıcıları listele
  /system [metin|reset]  bu oturumun sistem istemini göster/ayarla
  /set [ad değer]        üretim parametrelerini göster/ayarla ('reset' temizler)
  /regen                 son mesajınıza yeni bir yanıt üret
  /undo                  son karşılıklı mesajları geri al
  /search [--all] metin  mesajlarda ara (bu oturumda, --all ile hepsinde)
  /char list | new | use <ad> | show | edit | delete <ad>
                         karakterler: listele, oluştur (sihirbaz), sohbete başla, gör, düzenle, sil
  /char ai [first_mes|mes_example|description]
                         mevcut karakterin bir alanını yapay zekâya yazdır
  /char import <dosya>   topluluk kartı içe aktar (.json veya .png)
  /char export [ad]      karakteri .json kart olarak kaydet
  /persona [set|clear]   karakterlere kim olduğunuzu anlatın
  /free [list|all|N]     ücretsiz OpenRouter modellerini test et, en hızlısına geç
  /theme [ad|reset]      renk temasını göster/değiştir
  /clear                 ekranı temizle (geçmiş silinmez)
  /config                ayar ve veri dosyalarının konumunu göster
  /quit                  çıkış
Tuşlar: Enter gönder · Alt+Enter yeni satır · Ctrl+C yanıtı durdur · Ctrl+D çıkış""",
    },
}


def detect_language() -> str:
    for var in ("CHATCLIENT_LANG", "LC_ALL", "LC_MESSAGES", "LANG"):
        val = os.environ.get(var, "")
        if val:
            return "tr" if val.lower().startswith("tr") else "en"
    try:
        loc = locale.getlocale()[0] or ""
    except ValueError:
        loc = ""
    if not loc and os.name == "nt":
        try:
            import ctypes

            lcid = ctypes.windll.kernel32.GetUserDefaultUILanguage()  # type: ignore[attr-defined]
            return "tr" if (lcid & 0x3FF) == 0x1F else "en"  # 0x1F = LANG_TURKISH
        except Exception:
            pass
    return "tr" if loc.lower().startswith(("tr", "turkish")) else "en"


class Translator:
    def __init__(self, language: str = "auto"):
        self.lang = detect_language() if language == "auto" else language

    def __call__(self, key: str, **kwargs: object) -> str:
        entry = STRINGS.get(key)
        if entry is None:
            return key
        text = entry.get(self.lang) or entry["en"]
        return text.format(**kwargs) if kwargs else text
