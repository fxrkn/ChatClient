"""Interface strings (Turkish / English). The AI itself always answers in the user's language."""

from __future__ import annotations

import locale
import os
from dataclasses import dataclass

LANG_KEY = "ui_lang"  # language chosen with /lang (overrides config)

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
    "refusal_fallback": {
        "en": "{old} declined; {new} is writing this reply instead (the declined one: /prev).",
        "tr": "{old} reddetti; bu yanıtı {new} yazıyor (reddedilen sürüm: /prev).",
    },
    "rewrite_repeat": {
        "en": "That reply repeated {pct} of earlier ones; writing a fresh one (the repeat: /prev).",
        "tr": "Bu yanıt öncekilerin {pct} kadarını tekrarladı; yenisi yazılıyor (tekrar eden: /prev).",
    },
    "private_on": {
        "en": "Private session: nothing here is remembered, indexed or recalled elsewhere. "
        "What memory already took from it was deleted.",
        "tr": "Gizli oturum: buradaki hiçbir şey hatırlanmaz, indekslenmez, başka yerde "
        "anılmaz. Hafızanın bu oturumdan aldıkları silindi.",
    },
    "private_off": {
        "en": "Private mode off: memory works in this session again (from now on).",
        "tr": "Gizli mod kapalı: hafıza bu oturumda yeniden çalışıyor (bundan sonrası için).",
    },
    "content_mode": {
        "en": "Content: {mode}  (/content standard|mature)",
        "tr": "İçerik: {mode}  (/content standard|mature)",
    },
    "fallback_set": {
        "en": "Declined replies are rewritten by {model}.",
        "tr": "Reddedilen yanıtları {model} yeniden yazar.",
    },
    "fallback_off": {
        "en": "No fallback for declined replies (set one: /fallback provider:model).",
        "tr": "Reddedilen yanıtlar için yedek model yok (ayarla: /fallback sağlayıcı:model).",
    },
    "truncated_len": {
        "en": "[reply hit max_tokens — raise it with /set max_tokens N]",
        "tr": "[yanıt max_tokens sınırına ulaştı — /set max_tokens N ile artırın]",
    },
    "you": {"en": "You", "tr": "Sen"},
    "model_load": {"en": "model load {s}s", "tr": "model yükleme {s}s"},
    "typing": {"en": "{name} is typing…", "tr": "{name} yazıyor…"},
    "picker_footer": {
        "en": "↑↓ move · type to filter · Enter choose · Esc cancel",
        "tr": "↑↓ gez · yazarak filtrele · Enter seç · Esc vazgeç",
    },
    "pick_model": {"en": "Choose a model", "tr": "Model seçin"},
    "pick_session": {"en": "Switch to session", "tr": "Oturuma geç"},
    "pick_char": {"en": "Start a chat with", "tr": "Sohbete başla"},
    "no_models_found": {
        "en": "No models found. Is a local server running, or is an API key set (/key)?",
        "tr": "Model bulunamadı. Yerel sunucu açık mı, ya da API anahtarı ekli mi (/key)?",
    },
    "alt_hint": {
        "en": "Alt+←/→ (or Ctrl+←/→) other versions · → on the last writes a new one",
        "tr": "Alt+←/→ (veya Ctrl+←/→) diğer sürümler · sondayken → yenisini yazar",
    },
    "char_color_usage": {
        "en": "Usage: /char color #rrggbb   (or: /char color auto)",
        "tr": "Kullanım: /char color #rrggbb   (ya da: /char color auto)",
    },
    "char_color_set": {
        "en": "{name} now uses {color}",
        "tr": "{name} artık {color} rengini kullanıyor",
    },
    "key_usage": {
        "en": "Usage: /key set <provider>  (providers: {names})",
        "tr": "Kullanım: /key set <sağlayıcı>  (sağlayıcılar: {names})",
    },
    "key_unknown": {
        "en": "Unknown provider '{name}' (key providers: {names})",
        "tr": "Bilinmeyen sağlayıcı '{name}' (anahtar kullananlar: {names})",
    },
    "key_prompt": {
        "en": "API key for {name} (hidden, Enter to cancel): ",
        "tr": "{name} API anahtarı (gizli, iptal için boş Enter): ",
    },
    "key_checking": {"en": "Checking the key…", "tr": "Anahtar kontrol ediliyor…"},
    "key_rejected": {
        "en": "{name} rejected this key; nothing was saved. Copy it again and retry.",
        "tr": "{name} bu anahtarı reddetti; kaydedilmedi. Anahtarı yeniden kopyalayıp tekrar deneyin.",
    },
    "key_unverified": {
        "en": "Could not verify the key ({error}); saving it anyway.",
        "tr": "Anahtar doğrulanamadı ({error}); yine de kaydediliyor.",
    },
    "key_save_failed": {
        "en": "Could not save the key: {error}",
        "tr": "Anahtar kaydedilemedi: {error}",
    },
    "key_saved_keychain": {
        "en": "✓ Key for {name} saved in the OS keychain.",
        "tr": "✓ {name} anahtarı sistem anahtar deposuna kaydedildi.",
    },
    "key_saved_file": {
        "en": "✓ Key for {name} saved to {path} (readable only by you; no OS keychain found).",
        "tr": "✓ {name} anahtarı {path} dosyasına kaydedildi (yalnızca siz okuyabilirsiniz; sistem anahtar deposu bulunamadı).",
    },
    "key_env_override": {
        "en": "Note: the {env} environment variable is set and takes priority over the saved key.",
        "tr": "Not: {env} ortam değişkeni tanımlı ve kaydedilen anahtarın önüne geçiyor.",
    },
    "key_removed": {
        "en": "Removed the saved key for {name}.",
        "tr": "{name} için kayıtlı anahtar silindi.",
    },
    "key_nothing_removed": {
        "en": "No saved key for {name}.",
        "tr": "{name} için kayıtlı anahtar yok.",
    },
    "key_still_from": {
        "en": "A key is still provided by: {source}",
        "tr": "Anahtar hâlâ şuradan geliyor: {source}",
    },
    "key_missing": {
        "en": "No API key for {name}. Add one with: /key set {name}",
        "tr": "{name} için API anahtarı yok. Eklemek için: /key set {name}",
    },
    "key_list_hint": {
        "en": "/key set <provider> to add or replace · /key remove <provider>",
        "tr": "Eklemek/değiştirmek için /key set <sağlayıcı> · silmek için /key remove <sağlayıcı>",
    },
    "key_src_env": {"en": "environment variable", "tr": "ortam değişkeni"},
    "key_src_keychain": {"en": "OS keychain", "tr": "sistem anahtar deposu"},
    "key_src_file": {"en": "key file", "tr": "anahtar dosyası"},
    "key_src_config": {"en": "config.toml", "tr": "config.toml"},
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
        "en": "Who is {name}, in one line? (job, situation)",
        "tr": "{name} kim, tek satırda? (meslek, durum)",
    },
    "q_content": {
        "en": "What kind of character is {name}?",
        "tr": "{name} nasıl bir karakter?",
    },
    "q_age": {
        "en": "{name}'s age? (a number or your own words; Enter skips)",
        "tr": "{name} kaç yaşında? (sayı ya da kendi ifadeniz; Enter atlar)",
    },
    "q_age_mature": {
        "en": "{name}'s age? (18 or older)",
        "tr": "{name} kaç yaşında? (18 ya da üstü)",
    },
    "age_adult_needed": {
        "en": "A mature character needs an age of 18 or older, as a number.",
        "tr": "Yetişkin içerikli bir karakterin yaşı sayı olarak 18 ya da üstü olmalı.",
    },
    "q_gender": {"en": "{name} is a…", "tr": "{name} bir…"},
    "q_appearance": {
        "en": "What does {name} look like? (optional: build, hair, clothes, a detail)",
        "tr": "{name} nasıl görünüyor? (isteğe bağlı: yapı, saç, kıyafet, bir ayrıntı)",
    },
    "q_likes": {
        "en": "What does {name} like and dislike? (optional)",
        "tr": "{name} neleri sever, nelerden hoşlanmaz? (isteğe bağlı)",
    },
    "q_backstory": {
        "en": "{name}'s background, in a sentence or two? (optional)",
        "tr": "{name}'in geçmişi, bir iki cümleyle? (isteğe bağlı)",
    },
    "q_history": {
        "en": "What has happened between you and {name} so far? (optional)",
        "tr": "Sizinle {name} arasında şimdiye kadar neler oldu? (isteğe bağlı)",
    },
    "q_themes": {
        "en": "Which themes should the story explore?",
        "tr": "Hikâye hangi temaları işlesin?",
    },
    "q_save_char": {
        "en": "Save {name} like this? [Y/n]",
        "tr": "{name} bu hâliyle kaydedilsin mi? [E/h]",
    },
    "char_not_saved": {
        "en": "Not saved.",
        "tr": "Kaydedilmedi.",
    },
    "content_mode_char": {
        "en": "Content for {name}: {mode}  (/content standard|mature)",
        "tr": "{name} için içerik: {mode}  (/content standard|mature)",
    },
    "theme_sample_name": {"en": "Name", "tr": "İsim"},
    "theme_sample_text": {"en": "narration,", "tr": "anlatım,"},
    "theme_sample_action": {"en": "action", "tr": "eylem"},
    "theme_sample_speech": {"en": '"speech"', "tr": '"konuşma"'},
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
    "imported_character": {
        "en": "Imported {name}: {chats} chats, {memories} memories. /char {name} to start.",
        "tr": "{name} içe aktarıldı: {chats} sohbet, {memories} anı. Başlamak için /char {name}.",
    },
    "imported_persona": {
        "en": "Imported the persona {name}.",
        "tr": "{name} personası içe aktarıldı.",
    },
    "imported_chat": {
        "en": "Imported the chat “{name}” (/sessions).",
        "tr": "“{name}” sohbeti içe aktarıldı (/sessions).",
    },
    "imported_copy": {
        "en": " It was already here, so this is a copy.",
        "tr": " Zaten vardı, bu yüzden bir kopya oluşturuldu.",
    },
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
    "edit_q": {
        "en": "Edit your message (Enter sends, Alt+Enter new line, Ctrl+C cancels):",
        "tr": "Mesajınızı düzenleyin (Enter gönderir, Alt+Enter yeni satır, Ctrl+C iptal):",
    },
    "edit_reply_q": {
        "en": "Edit the reply (Enter saves, Alt+Enter new line, Ctrl+C cancels):",
        "tr": "Yanıtı düzenleyin (Enter kaydeder, Alt+Enter yeni satır, Ctrl+C iptal):",
    },
    "edit_no_message": {
        "en": "There is no message of yours to edit.",
        "tr": "Düzenlenecek mesajınız yok.",
    },
    "edit_no_reply": {"en": "There is no reply yet.", "tr": "Henüz bir yanıt yok."},
    "edit_reply_saved": {
        "en": "Saved your edit as a new version (Alt+← shows the original).",
        "tr": "Düzenlemeniz yeni bir sürüm olarak kaydedildi (Alt+← orijinali gösterir).",
    },
    "ooc_usage": {
        "en": "Usage: /ooc <message>   e.g. /ooc slow the story down a bit",
        "tr": "Kullanım: /ooc <mesaj>   örn. /ooc hikâyeyi biraz yavaşlat",
    },
    "copied": {"en": "Copied to the clipboard.", "tr": "Panoya kopyalandı."},
    "copied_terminal": {
        "en": "Sent to the clipboard via the terminal (if nothing was copied, your terminal doesn't support it).",
        "tr": "Terminal üzerinden panoya gönderildi (kopyalanmadıysa terminaliniz bunu desteklemiyor).",
    },
    "export_empty": {"en": "Nothing to export yet.", "tr": "Henüz dışa aktarılacak bir şey yok."},
    "export_usage": {"en": "Usage: /export [{formats}]", "tr": "Kullanım: /export [{formats}]"},
    "exported": {"en": "Saved → {path}", "tr": "Kaydedildi → {path}"},
    "lang_usage": {"en": "Usage: /lang en | tr | auto", "tr": "Kullanım: /lang en | tr | auto"},
    "lang_set": {"en": "Interface language: English", "tr": "Arayüz dili: Türkçe"},
    "palette_title": {"en": "Commands", "tr": "Komutlar"},
    "draft_restored": {
        "en": "Your unsent message was restored.",
        "tr": "Gönderilmemiş mesajınız geri yüklendi.",
    },
    "setup_title": {"en": "Welcome to ChatClient!", "tr": "ChatClient'a hoş geldiniz!"},
    "setup_intro": {
        "en": "Let's get you chatting in a minute. You can rerun this anytime with /setup.",
        "tr": "Bir dakikada sohbete başlayalım. Bunu istediğiniz zaman /setup ile tekrar açabilirsiniz.",
    },
    "setup_how": {"en": "Where should the AI run?", "tr": "Yapay zekâ nerede çalışsın?"},
    "setup_local": {"en": "On this computer", "tr": "Bu bilgisayarda"},
    "setup_local_hint": {
        "en": "free, private, works offline (needs Ollama or LM Studio)",
        "tr": "ücretsiz, gizli, çevrimdışı çalışır (Ollama ya da LM Studio gerekir)",
    },
    "setup_cloud": {"en": "Online, free models", "tr": "İnternetten, ücretsiz modeller"},
    "setup_cloud_hint": {
        "en": "no install; needs a free OpenRouter account",
        "tr": "kurulum yok; ücretsiz bir OpenRouter hesabı gerekir",
    },
    "setup_skip": {"en": "Skip for now", "tr": "Şimdilik geç"},
    "setup_looking": {
        "en": "Looking for a local AI server…",
        "tr": "Yerel yapay zekâ sunucusu aranıyor…",
    },
    "setup_no_local": {
        "en": "No local server with a model was found.",
        "tr": "Model yüklü bir yerel sunucu bulunamadı.",
    },
    "setup_install_ollama": {
        "en": "Install Ollama from https://ollama.com/download, then run:  ollama pull {model}",
        "tr": "Ollama'yı https://ollama.com/download adresinden kurun, sonra şunu çalıştırın:  ollama pull {model}",
    },
    "setup_retry_q": {"en": "Look again? [Y/n]", "tr": "Tekrar bakılsın mı? [E/h]"},
    "setup_key_info": {
        "en": "Create a free key at https://openrouter.ai/keys and paste it below.",
        "tr": "https://openrouter.ai/keys adresinden ücretsiz bir anahtar oluşturup aşağıya yapıştırın.",
    },
    "setup_your_name": {
        "en": "What should characters call you? (optional)",
        "tr": "Karakterler size ne diye hitap etsin? (isteğe bağlı)",
    },
    "setup_pick_char_q": {
        "en": "Start a chat with a character now? [Y/n]",
        "tr": "Şimdi bir karakterle sohbete başlansın mı? [E/h]",
    },
    "setup_create_char_q": {
        "en": "Create your first character now? [Y/n]",
        "tr": "İlk karakterinizi şimdi oluşturalım mı? [E/h]",
    },
    "setup_done": {
        "en": "All set. Type a message, or press Ctrl+P to see what you can do.",
        "tr": "Hazır. Bir mesaj yazın ya da neler yapabileceğinizi görmek için Ctrl+P'ye basın.",
    },
    "recalled_n": {"en": "remembered {n}", "tr": "{n} anı"},
    "conflict_noted": {"en": "⚠ {n} contradiction noted", "tr": "⚠ {n} çelişki fark edildi"},
    "conflicts_none": {"en": "No contradictions found.", "tr": "Çelişki bulunmadı."},
    "memory_outdated": {"en": "outdated", "tr": "eskidi"},
    "semantic_status": {
        "en": "Meaning-based recall: {state} {detail}",
        "tr": "Anlam tabanlı hatırlama: {state} {detail}",
    },
    "semantic_off": {"en": "off", "tr": "kapalı"},
    "semantic_waiting": {"en": "starting soon", "tr": "birazdan başlayacak"},
    "semantic_loading": {
        "en": "loading the model (first time: downloading ~0.2 GB)",
        "tr": "model yükleniyor (ilk sefer: ~0,2 GB indiriliyor)",
    },
    "semantic_ready": {"en": "ready", "tr": "hazır"},
    "semantic_failed": {"en": "unavailable (see the log)", "tr": "kullanılamıyor (günlüğe bakın)"},
    "semantic_waiting_n": {"en": "{n} waiting", "tr": "{n} bekliyor"},
    "memory_off": {
        "en": "Memory is turned off ([memory] enabled = false in the config).",
        "tr": "Hafıza kapalı (ayar dosyasında [memory] enabled = false).",
    },
    "memory_none": {"en": "Nothing remembered yet.", "tr": "Henüz hatırlanan bir şey yok."},
    "memory_hint": {
        "en": "/memory pin|unpin|delete <#> · /memory add <text> · /memory <search> · /memory status",
        "tr": "/memory pin|unpin|delete <#> · /memory add <metin> · /memory <arama> · /memory status",
    },
    "memory_add_usage": {
        "en": "Usage: /memory add <something to remember>",
        "tr": "Kullanım: /memory add <hatırlanacak şey>",
    },
    "memory_added": {"en": "Remembered (#{id}).", "tr": "Hatırlanacak (#{id})."},
    "memory_id_usage": {
        "en": "Usage: /memory {cmd} <number>",
        "tr": "Kullanım: /memory {cmd} <numara>",
    },
    "memory_not_found": {"en": "No memory #{id}.", "tr": "#{id} numaralı anı yok."},
    "memory_pinned": {
        "en": "Pinned #{id}: always kept in mind.",
        "tr": "#{id} sabitlendi: hep akılda tutulacak.",
    },
    "memory_unpinned": {"en": "Unpinned #{id}.", "tr": "#{id} sabitlemesi kaldırıldı."},
    "memory_deleted": {"en": "Forgot #{id}.", "tr": "#{id} unutuldu."},
    "memory_running": {"en": "Updating memory now…", "tr": "Hafıza şimdi güncelleniyor…"},
    "memory_ran": {"en": "Done ({n} job(s)).", "tr": "Tamamlandı ({n} iş)."},
    "memory_last_error": {"en": "Last memory error: {error}", "tr": "Son hafıza hatası: {error}"},
    "memory_status": {
        "en": "{memories} memories ({scope}) · {segments} summaries · {days} day digests · {pending} jobs waiting · {failed} failed",
        "tr": "{memories} anı ({scope}) · {segments} özet · {days} gün özeti · {pending} bekleyen iş · {failed} başarısız",
    },
    "scope_character": {
        "en": "shared across this character's chats",
        "tr": "bu karakterin tüm sohbetlerinde ortak",
    },
    "scope_session": {"en": "this chat only", "tr": "yalnızca bu sohbet"},
    "recall_usage": {
        "en": "Usage: /recall <what to look for>   e.g. /recall what did we do yesterday",
        "tr": "Kullanım: /recall <ne aranacak>   örn. /recall dün ne yaptık",
    },
    "recall_date": {"en": "dates: {range}", "tr": "tarihler: {range}"},
    "recall_no_brief": {
        "en": "(no brief: the utility model is off or unavailable; showing search results)",
        "tr": "(brif yok: yardımcı model kapalı ya da ulaşılamıyor; arama sonuçları gösteriliyor)",
    },
    "scene_none": {
        "en": "No scene recorded yet (it updates as you chat).",
        "tr": "Henüz sahne kaydı yok (sohbet ettikçe güncellenir).",
    },
    "scene_cleared": {"en": "Scene cleared.", "tr": "Sahne temizlendi."},
    "summary_none": {
        "en": "No summaries yet (they appear as the chat grows).",
        "tr": "Henüz özet yok (sohbet uzadıkça oluşur).",
    },
    "summary_story": {"en": "Story so far", "tr": "Hikâye şimdiye kadar"},
    "day_usage": {
        "en": "Usage: /day yesterday | /day 12 October | /day last week",
        "tr": "Kullanım: /day dün | /day 12 Ekim | /day geçen hafta",
    },
    "day_none": {"en": "No day summary for {days} yet.", "tr": "{days} için henüz gün özeti yok."},
    "help_title": {"en": "Commands", "tr": "Komutlar"},
    "help_keys": {
        "en": "Keys: Enter send · Alt+Enter new line · Ctrl+P command palette · Alt+←/→ reply versions · Ctrl+C stop a reply · Ctrl+D quit",
        "tr": "Tuşlar: Enter gönder · Alt+Enter yeni satır · Ctrl+P komut paleti · Alt+←/→ yanıt sürümleri · Ctrl+C yanıtı durdur · Ctrl+D çıkış",
    },
}


@dataclass(frozen=True)
class CommandDoc:
    command: str  # what the palette inserts / runs
    args: str  # usage hint after the command ("" = runs directly from the palette)
    en: str
    tr: str
    args_tr: str = ""  # Turkish usage hint, if it differs

    def usage(self, lang: str) -> str:
        args = self.args_tr if lang == "tr" and self.args_tr else self.args
        return f"{self.command} {args}".strip()

    def describe(self, lang: str) -> str:
        return self.tr if lang == "tr" else self.en


# Single source for /help and the Ctrl+P palette.
COMMAND_DOCS: tuple[CommandDoc, ...] = (
    CommandDoc("/new", "[title]", "start a new session", "yeni oturum başlat", "[başlık]"),
    CommandDoc("/sessions", "", "list sessions", "oturumları listele"),
    CommandDoc("/switch", "", "switch session (pick from a list, or /switch <id>)", "oturuma geç (listeden seç ya da /switch <id>)"),
    CommandDoc("/rename", "<title>", "rename this session", "bu oturumu yeniden adlandır", "<başlık>"),
    CommandDoc("/delete", "<id>", "delete a session", "oturum sil"),
    CommandDoc("/history", "[n]", "show the last n messages", "son n mesajı göster"),
    CommandDoc("/edit", "", "edit your last message and resend it (/edit reply: fix the AI's)", "son mesajınızı düzenleyip yeniden gönderin (/edit reply: AI yanıtını düzeltin)"),
    CommandDoc("/continue", "", "let the AI carry on from where its reply stopped", "yanıtı kaldığı yerden sürdürt"),
    CommandDoc("/regen", "", "write a new version of the last reply (/regen <model>: with another model)", "son yanıtın yeni bir sürümünü yazdır (/regen <model>: başka modelle)"),
    CommandDoc("/prev", "", "previous version of the last reply (Alt+←)", "son yanıtın önceki sürümü (Alt+←)"),
    CommandDoc("/next", "", "next version of the last reply (Alt+→)", "son yanıtın sonraki sürümü (Alt+→)"),
    CommandDoc("/undo", "", "remove the last exchange", "son karşılıklı mesajları geri al"),
    CommandDoc("/ooc", "<message>", "talk to the AI out of character", "karakter dışı (OOC) konuş", "<mesaj>"),
    CommandDoc("/copy", "", "copy the last reply (/copy all: the whole chat)", "son yanıtı kopyala (/copy all: tüm sohbet)"),
    CommandDoc("/export", "", "save the chat as md, html, txt or jsonl (jsonl: re-importable)", "sohbeti md, html, txt ya da jsonl olarak kaydet (jsonl: geri yüklenebilir)"),
    CommandDoc("/import", "<file>", "import a character (.zip/.json/.png), a persona or a chat (.jsonl)", "karakter (.zip/.json/.png), persona ya da sohbet (.jsonl) içe aktar", "<dosya>"),
    CommandDoc("/open", "", "open a folder: a character's, or data, characters, personas, plain-chats, exports", "klasör aç: karakterinki ya da data, characters, personas, plain-chats, exports"),
    CommandDoc("/search", "<text>", "search messages (/search --all: every session)", "mesajlarda ara (/search --all: tüm oturumlar)", "<metin>"),
    CommandDoc("/model", "", "change the model (pick from a list, or /model provider:model)", "modeli değiştir (listeden seç ya da /model sağlayıcı:model)"),
    CommandDoc("/models", "", "list available models", "kullanılabilir modelleri listele"),
    CommandDoc("/free", "", "find a working free OpenRouter model", "çalışan ücretsiz bir OpenRouter modeli bul"),
    CommandDoc("/providers", "", "list configured providers", "tanımlı sağlayıcıları listele"),
    CommandDoc("/key", "", "show, add or remove API keys (/key set <provider>)", "API anahtarlarını göster, ekle, sil (/key set <sağlayıcı>)"),
    CommandDoc("/char list", "", "list characters", "karakterleri listele"),
    CommandDoc("/char new", "", "create a character (wizard)", "karakter oluştur (sihirbaz)"),
    CommandDoc("/char use", "", "start a chat with a character", "bir karakterle sohbete başla"),
    CommandDoc("/char show", "", "show the current character's card", "mevcut karakterin kartını göster"),
    CommandDoc("/char edit", "", "edit the current character", "mevcut karakteri düzenle"),
    CommandDoc("/char color", "<#rrggbb|auto>", "set the character's name colour", "karakterin isim rengini ayarla"),
    CommandDoc("/char ai", "<field>", "let the AI rewrite a field (first_mes, mes_example, description)", "bir alanı AI'a yazdır (first_mes, mes_example, description)", "<alan>"),
    CommandDoc("/char import", "<file>", "import a character (.zip with chats & memory, or a .json/.png card)", "karakter içe aktar (sohbet ve hafızalı .zip ya da .json/.png kart)", "<dosya>"),
    CommandDoc("/char export", "", "save a character as .zip with chats & memory (/char export <name> card: the card only)", "karakteri sohbet ve hafızasıyla .zip olarak kaydet (/char export <ad> card: yalnızca kart)"),
    CommandDoc("/char delete", "<name>", "delete a character", "karakter sil", "<ad>"),
    CommandDoc("/memory", "", "what the AI remembers: list, search, pin, delete, add", "AI'ın hatırladıkları: listele, ara, sabitle, sil, ekle"),
    CommandDoc("/recall", "<text>", "test what would be remembered, e.g. /recall yesterday", "neyin hatırlanacağını dene, örn. /recall dün", "<metin>"),
    CommandDoc("/day", "", "what happened on a day (/day yesterday, /day 12 October)", "bir günde ne oldu (/day dün, /day 12 Ekim)"),
    CommandDoc("/summary", "", "the story so far", "hikâye şimdiye kadar"),
    CommandDoc("/scene", "", "current scene: place, time, mood, relationship", "mevcut sahne: yer, zaman, ruh hâli, ilişki"),
    CommandDoc("/persona", "", "tell characters who you are (/persona set · export · import <file>)", "karakterlere kim olduğunuzu anlatın (/persona set · export · import <dosya>)"),
    CommandDoc("/system", "", "show or set this session's system prompt", "bu oturumun sistem istemini göster/ayarla"),
    CommandDoc("/set", "", "show or set generation parameters (/set temperature 0.9)", "üretim parametrelerini göster/ayarla (/set temperature 0.9)"),
    CommandDoc("/theme", "", "show or switch the colour theme", "renk temasını göster/değiştir"),
    CommandDoc("/private", "[on|off]", "private session: no memory, no indexing", "gizli oturum: hafıza ve indeksleme yok"),
    CommandDoc("/content", "[standard|mature]", "content setting (mature: adult fiction, no lectures)", "içerik ayarı (mature: yetişkin kurgu, ahlak dersi yok)"),
    CommandDoc("/fallback", "[provider:model|off]", "model that rewrites declined replies", "reddedilen yanıtları yeniden yazan model"),
    CommandDoc("/lang", "<en|tr|auto>", "interface language", "arayüz dili"),
    CommandDoc("/setup", "", "run the first-time setup again", "ilk kurulum sihirbazını yeniden çalıştır"),
    CommandDoc("/clear", "", "clear the screen (history is kept)", "ekranı temizle (geçmiş silinmez)"),
    CommandDoc("/config", "", "show where config and data files are", "ayar ve veri dosyalarının yerini göster"),
    CommandDoc("/help", "", "show this help", "bu yardımı göster"),
    CommandDoc("/quit", "", "exit", "çıkış"),
)  # fmt: skip


def help_text(t: Translator, width: int = 24) -> str:
    lines = [t("help_title")]
    for doc in COMMAND_DOCS:
        usage = doc.usage(t.lang)
        if len(usage) < width - 2:
            lines.append(f"  {usage:<{width - 2}} {doc.describe(t.lang)}")
        else:
            lines.append(f"  {usage}")
            lines.append(f"  {'':<{width - 2}} {doc.describe(t.lang)}")
    lines.append(t("help_keys"))
    return "\n".join(lines)


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
        self.set_language(language)

    def set_language(self, language: str) -> None:
        self.lang = detect_language() if language == "auto" else language

    def __call__(self, key: str, **kwargs: object) -> str:
        entry = STRINGS.get(key)
        if entry is None:
            return key
        text = entry.get(self.lang) or entry["en"]
        return text.format(**kwargs) if kwargs else text
