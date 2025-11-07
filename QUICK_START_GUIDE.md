# Быстрый старт: Obsidian to Anki APF Sync

Руководство по использованию системы синхронизации заметок Obsidian с карточками Anki в формате APF.

---

## Как запустить синхронизацию

### Вариант 1: Тестовый запуск (рекомендуется для начала)

Обработает случайные 5 заметок в dry-run режиме (без записи в Anki):

```bash
cd /Users/npochaev/GitHub/obsidian-to-anki
uv run obsidian-anki-sync test-run --count 5 --use-agents --config config.yaml
```

### Вариант 2: Сухой прогон (preview)

Покажет, что будет сделано, без реального изменения Anki:

```bash
uv run obsidian-anki-sync sync --dry-run --use-agents --config config.yaml
```

### Вариант 3: Реальная синхронизация

**⚠️ Внимание**: Это создаст реальные карточки в Anki!

```bash
uv run obsidian-anki-sync sync --use-agents --config config.yaml
```

### Вариант 4: Инкрементальная синхронизация

Обработает только новые заметки:

```bash
uv run obsidian-anki-sync sync --incremental --use-agents --config config.yaml
```

---

## Полезные команды

### Информационные команды

```bash
# Посмотреть колоды Anki
uv run obsidian-anki-sync decks --config config.yaml

# Посмотреть модели карточек
uv run obsidian-anki-sync models --config config.yaml

# Посмотреть поля модели
uv run obsidian-anki-sync model-fields --model "APF: Simple (3.0.0)" --config config.yaml

# Валидировать одну заметку
uv run obsidian-anki-sync validate path/to/note.md --config config.yaml

# Показать индекс (статистика по заметкам и карточкам)
uv run obsidian-anki-sync index --config config.yaml

# Показать прогресс синхронизации
uv run obsidian-anki-sync progress --config config.yaml

# Инициализировать конфигурацию и базу данных
uv run obsidian-anki-sync init --config config.yaml
```

### Команды синхронизации

```bash
# Полная синхронизация (создаёт/обновляет все карточки)
uv run obsidian-anki-sync sync --use-agents --config config.yaml

# Сухой прогон (показывает, что будет сделано)
uv run obsidian-anki-sync sync --dry-run --use-agents --config config.yaml

# Инкрементальная синхронизация (только новые заметки)
uv run obsidian-anki-sync sync --incremental --use-agents --config config.yaml

# Синхронизация без индексации (не рекомендуется)
uv run obsidian-anki-sync sync --no-index --use-agents --config config.yaml

# Возобновление прерванной синхронизации
uv run obsidian-anki-sync sync --resume <session-id> --use-agents --config config.yaml

# Отключить автоматическое возобновление
uv run obsidian-anki-sync sync --no-resume --use-agents --config config.yaml
```

### Тестовые команды

```bash
# Тестовый запуск на N случайных заметках
uv run obsidian-anki-sync test-run --count 5 --use-agents --config config.yaml

# Тестовый запуск на 10 заметках
uv run obsidian-anki-sync test-run --count 10 --use-agents --config config.yaml
```

### Экспорт в .apkg файл

Создание автономного файла для импорта в Anki:

```bash
# Экспортировать 10 случайных заметок (для тестирования)
uv run obsidian-anki-sync export --sample 10 --use-agents --config config.yaml

# Экспортировать все заметки
uv run obsidian-anki-sync export --use-agents --config config.yaml --output my-cards.apkg

# Экспортировать с кастомным именем колоды
uv run obsidian-anki-sync export --use-agents --config config.yaml \
  --deck-name "My Custom Deck" \
  --deck-description "Exported from Obsidian" \
  --output my-cards.apkg
```

### Управление прогрессом

```bash
# Показать все сессии синхронизации
uv run obsidian-anki-sync progress --config config.yaml

# Удалить все завершённые сессии
uv run obsidian-anki-sync clean-progress --all-completed --config config.yaml

# Удалить конкретную сессию
uv run obsidian-anki-sync clean-progress --session <session-id> --config config.yaml
```

### Отладка и логирование

```bash
# Увеличить детализацию логов
uv run obsidian-anki-sync sync --use-agents --config config.yaml --log-level DEBUG

# Показать только ошибки
uv run obsidian-anki-sync sync --use-agents --config config.yaml --log-level ERROR

# Валидация с подробным выводом
uv run obsidian-anki-sync validate path/to/note.md --config config.yaml --log-level DEBUG
```

---

## Настройка `config.yaml`

### Основные параметры

```yaml
# Путь к хранилищу Obsidian
vault_path: "~/Documents/InterviewQuestions"
source_dir: "InterviewQuestions"

# Настройки Anki
anki_connect_url: "http://127.0.0.1:8765"
anki_deck_name: "Interview Questions"
anki_note_type: "APF: Simple (3.0.0)"

# База данных состояния
db_path: ".sync_state.db"
```

### Мультиагентная система

```yaml
# Использовать ли мультиагентную систему
use_agent_system: true

# Режим выполнения агентов
# - parallel: быстрее, но использует 30-34GB RAM
# - sequential: медленнее, но меньше RAM (15-20GB)
agent_execution_mode: "parallel"

# Настройки Ollama
ollama_base_url: "http://localhost:11434"

# Модели для агентов
pre_validator_model: "qwen3:8b" # Быстрая валидация структуры
generator_model: "qwen3:32b" # Генерация карточек
post_validator_model: "qwen3:14b" # Проверка качества

# Температура для генерации (0.0-1.0)
pre_validator_temperature: 0.0
generator_temperature: 0.3
post_validator_temperature: 0.0

# Настройки валидации
pre_validation_enabled: true
post_validation_max_retries: 3
post_validation_auto_fix: true
post_validation_strict_mode: true
```

### Режимы работы

```yaml
# Режим работы
# - apply: реально создавать/обновлять карточки
# - dry-run: только показать, что будет сделано
run_mode: "apply"

# Что делать с удалёнными заметками
# - delete: удалять карточки из Anki
# - archive: помечать как архивные
delete_mode: "delete"

# Уровень логирования (DEBUG, INFO, WARN, ERROR)
log_level: "INFO"
```

### Альтернативные провайдеры LLM

**OpenRouter (облачный)**:

```yaml
use_agent_system: false
llm_provider: "openrouter"
openrouter_api_key: "sk-or-v1-..." # Или через OPENROUTER_API_KEY env var
openrouter_model: "openai/gpt-4"
```

**LM Studio (локальный GUI)**:

```yaml
use_agent_system: false
llm_provider: "lm_studio"
lm_studio_base_url: "http://localhost:1234/v1"
```

### Настройки экспорта

```yaml
# Настройки экспорта в .apkg
export_deck_name: "Interview Questions"
export_deck_description: "Generated from Obsidian notes"
export_output_path: "interview_questions.apkg"
```

---

## Структура заметок Obsidian

### Требования к файлам

-   **Имя файла**: должно начинаться с префикса `q-` (например, `q-microservices-vs-monolith--system-design--hard.md`)
-   **Расширение**: `.md`
-   **Расположение**: в директориях, соответствующих теме (например, `30-System-Design`, `40-Android`)

### YAML Frontmatter

Каждая заметка должна начинаться с YAML frontmatter между `---`:

```yaml
---
id: unique-id-here
title: "Question Title"
aliases: []
topic: system-design
subtopics: [microservices, architecture]
question_kind: conceptual
difficulty: hard
original_language: en
language_tags: [en, ru]
source: "Book/Article Name"
source_note: ""
status: active
moc: []
related: []
created: 2025-01-01
updated: 2025-01-15
tags: [system-design, microservices]
---
```

**Обязательные поля**: `id`, `title`, `topic`, `language_tags`, `created`, `updated`

### Формат Q/A блоков

Каждая пара вопрос-ответ оформляется следующим образом:

```markdown
# Question (EN)

> What is the difference between microservices and monolith?

# Вопрос (RU)

> В чём разница между микросервисами и монолитом?

---

## Answer (EN)

Microservices architecture splits an application into small, independent services...

## Ответ (RU)

Архитектура микросервисов разделяет приложение на небольшие независимые сервисы...

## Follow-ups

-   How do services communicate?
-   What are the trade-offs?

## References

-   [Link to resource](https://example.com)

## Related Questions

-   [[q-service-mesh--system-design--medium]]
```

**Важно**:

-   Разделитель `---` обязателен между вопросом и ответом
-   Одна заметка может содержать несколько Q/A блоков
-   Секции `Follow-ups`, `References`, `Related Questions` опциональны

### Проверка структуры заметки

```bash
# Валидировать одну заметку
uv run obsidian-anki-sync validate path/to/note.md --config config.yaml

# Валидировать все заметки (через sync с dry-run)
uv run obsidian-anki-sync sync --dry-run --config config.yaml
```

---

## Рекомендации для первого запуска

### Шаг 1: Тестовый запуск на малой выборке

Начните с test-run на 5 заметках:

```bash
uv run obsidian-anki-sync test-run --count 5 --use-agents --config config.yaml
```

**Что происходит:**

-   Система выберет 5 случайных заметок
-   Запустит мультиагентную обработку (Pre-Validator → Generator → Post-Validator)
-   Покажет результаты БЕЗ создания карточек в Anki
-   Процесс займёт 5-10 минут (LLM генерация требует времени)

### Шаг 2: Проверьте результаты

Если всё прошло успешно, вы увидите таблицу с результатами:

```text
┌─────────────────┬────────┐
│ Metric          │ Value  │
├─────────────────┼────────┤
│ notes_processed │ 5      │
│ cards_created   │ 10     │ (2 языка × 5 заметок)
│ cards_updated   │ 0      │
│ errors          │ 0      │
└─────────────────┴────────┘
```

### Шаг 3: Dry-run на всех заметках (опционально)

Если хотите посмотреть, что произойдёт со всеми заметками:

```bash
uv run obsidian-anki-sync sync --dry-run --use-agents --config config.yaml
```

**⚠️ Внимание**: Это займёт много времени (несколько часов), так как обработает все заметки. Рекомендуется сначала протестировать на малой выборке.

### Шаг 4: Реальная синхронизация

Когда убедитесь, что всё работает корректно:

```bash
uv run obsidian-anki-sync sync --use-agents --config config.yaml
```

**Это создаст реальные карточки в Anki!**

---

## Рабочий процесс после первого запуска

### Добавление новых заметок

После того, как вы создали новые заметки в Obsidian:

```bash
# Синхронизировать только новые заметки (рекомендуется)
uv run obsidian-anki-sync sync --incremental --use-agents --config config.yaml
```

### Обновление существующих заметок

Если вы изменили существующие заметки:

```bash
# Полная синхронизация (обновит изменённые карточки)
uv run obsidian-anki-sync sync --use-agents --config config.yaml
```

### Возобновление прерванной синхронизации

Если синхронизация была прервана:

```bash
# Система автоматически предложит возобновить
uv run obsidian-anki-sync sync --use-agents --config config.yaml

# Или принудительно возобновить конкретную сессию
uv run obsidian-anki-sync sync --resume <session-id> --use-agents --config config.yaml
```

---

## Альтернативные режимы работы

### Использование OpenRouter вместо локальных моделей

Если вы хотите использовать облачные LLM (быстрее, но требует API ключ):

1. Получите API key на <https://openrouter.ai>

2. Измените в `config.yaml`:

    ```yaml
    use_agent_system: false
    llm_provider: "openrouter"
    openrouter_api_key: "sk-or-v1-..." # Или через переменную окружения
    openrouter_model: "openai/gpt-4"
    ```

3. Запустите синхронизацию БЕЗ флага `--use-agents`:
    ```bash
    uv run obsidian-anki-sync sync --config config.yaml
    ```

### Использование LM Studio

Если вы предпочитаете GUI для управления моделями:

1. Скачайте LM Studio: <https://lmstudio.ai>

2. Загрузите модели через GUI

3. Запустите локальный сервер в LM Studio

4. Измените в `config.yaml`:
    ```yaml
    llm_provider: "lm_studio"
    lm_studio_base_url: "http://localhost:1234/v1"
    use_agent_system: false # LM Studio не поддерживает мультиагентную систему
    ```

---

## Мониторинг и отладка

### Проверка индекса

Посмотреть статистику по заметкам и карточкам:

```bash
uv run obsidian-anki-sync index --config config.yaml
```

### Просмотр прогресса синхронизации

```bash
uv run obsidian-anki-sync progress --config config.yaml
```

### Увеличение детализации логов

Для отладки проблем:

```bash
uv run obsidian-anki-sync sync --use-agents --config config.yaml --log-level DEBUG
```

### Очистка истории прогресса

```bash
# Удалить завершённые сессии
uv run obsidian-anki-sync clean-progress --all-completed --config config.yaml

# Удалить конкретную сессию
uv run obsidian-anki-sync clean-progress --session <session-id> --config config.yaml
```

---

## Устранение неполадок

### Проблема: Ollama не отвечает

**Симптом**: Ошибка `Connection refused` или `Failed to connect to Ollama`

**Решение**:

```bash
# Проверить, запущен ли Ollama
ps aux | grep ollama

# Если не запущен, запустить
ollama serve

# Проверить доступность
curl http://localhost:11434/api/tags
```

### Проблема: AnkiConnect не отвечает

**Симптом**: Ошибка `Failed to connect to AnkiConnect`

**Решение**:

1. Убедитесь, что Anki запущен
2. Проверьте, что AnkiConnect установлен: Tools → Add-ons
3. Проверьте доступность:
    ```bash
    curl -X POST http://127.0.0.1:8765 -d '{"action":"version","version":6}'
    ```

### Проблема: Модели не найдены

**Симптом**: Ошибка `Model not found`

**Решение**:

```bash
# Проверить установленные модели
ollama list

# Установить недостающие модели
ollama pull qwen3:8b
ollama pull qwen3:14b
ollama pull qwen3:32b
```

### Проблема: Недостаточно памяти

**Симптом**: Система зависает или выдаёт ошибку памяти

**Решение**:

```yaml
# Переключитесь на sequential режим
agent_execution_mode: "sequential"

# Или используйте меньшие модели
generator_model: "qwen3:14b" # Вместо qwen3:32b
```

### Проблема: База данных повреждена

**Симптом**: Ошибки при чтении `.sync_state.db`

**Решение**:

```bash
# Удалите базу данных (она будет пересоздана)
rm .sync_state.db

# Или переименуйте для резервной копии
mv .sync_state.db .sync_state.db.backup

# Запустите синхронизацию заново
uv run obsidian-anki-sync sync --use-agents --config config.yaml
```

### Проблема: Карточки создаются неправильно

**Симптом**: Карточки в Anki содержат ошибки или неполную информацию

**Решение**:

1. Проверьте структуру заметки: `uv run obsidian-anki-sync validate path/to/note.md`
2. Удалите неправильную карточку в Anki
3. Исправьте заметку в Obsidian
4. Запустите синхронизацию снова
5. Если проблема повторяется, включите DEBUG логирование

### Проблема: Синхронизация прерывается

**Симптом**: Процесс останавливается на середине

**Решение**:

```bash
# Проверьте незавершённые сессии
uv run obsidian-anki-sync progress --config config.yaml

# Возобновите последнюю сессию
uv run obsidian-anki-sync sync --resume <session-id> --use-agents --config config.yaml

# Или запустите инкрементальную синхронизацию (пропустит уже обработанные)
uv run obsidian-anki-sync sync --incremental --use-agents --config config.yaml
```

---

## Производительность

### Ожидаемая скорость обработки

На MacBook M4 Max (48GB RAM):

| Режим              | Скорость            | Использование RAM | Рекомендации        |
| ------------------ | ------------------- | ----------------- | ------------------- |
| Parallel agents    | ~10-15 карточек/мин | 30-34GB           | Для больших объёмов |
| Sequential agents  | ~5-8 карточек/мин   | 15-20GB           | При нехватке памяти |
| OpenRouter (cloud) | ~15-20 карточек/мин | <2GB              | Быстрый старт       |

### Оценка времени

Для расчёта времени умножьте количество заметок на количество языков (обычно 2):

-   **Parallel mode**: ~2-3 часа на 1000 заметок
-   **Sequential mode**: ~4-6 часов на 1000 заметок
-   **OpenRouter**: ~1.5-2 часа на 1000 заметок (+ стоимость API)

### Оптимизация производительности

```yaml
# Используйте parallel режим, если достаточно RAM
agent_execution_mode: "parallel"

# Или sequential для экономии памяти
agent_execution_mode: "sequential"

# Для быстрой обработки используйте инкрементальный режим
# (обрабатывает только новые заметки)
```

---

## Стоимость (только для OpenRouter)

| Модель            | Стоимость входа   | Стоимость выхода  | ~Стоимость на карточку |
| ----------------- | ----------------- | ----------------- | ---------------------- |
| GPT-4             | $0.03/1K tokens   | $0.06/1K tokens   | ~$0.05-0.10            |
| Claude 3.5 Sonnet | $0.003/1K tokens  | $0.015/1K tokens  | ~$0.02-0.04            |
| GPT-3.5 Turbo     | $0.0005/1K tokens | $0.0015/1K tokens | ~$0.01-0.02            |

**Пример**: Для 1000 карточек (2000 карточек с двумя языками):

-   GPT-4: ~$100-200
-   Claude 3.5 Sonnet: ~$40-80
-   GPT-3.5 Turbo: ~$20-40
-   Локальные модели (Ollama): $0 (после загрузки моделей)

---

## Автоматизация

### Регулярный запуск через cron

Добавьте в crontab для ежедневной синхронизации:

```bash
# Редактировать crontab
crontab -e

# Добавить строку (синхронизация каждый день в 2:00)
0 2 * * * cd /Users/npochaev/GitHub/obsidian-to-anki && uv run obsidian-anki-sync sync --incremental --use-agents --config config.yaml >> sync.log 2>&1
```

### Shell-скрипт для синхронизации

Создайте файл `sync.sh`:

```bash
#!/bin/bash
cd /Users/npochaev/GitHub/obsidian-to-anki
uv run obsidian-anki-sync sync --incremental --use-agents --config config.yaml
```

Сделайте исполняемым:

```bash
chmod +x sync.sh
./sync.sh
```

### Интеграция с Obsidian

Используйте плагин [Obsidian ShellCommands](https://obsidian.md/plugins?id=shellcommands) для запуска синхронизации прямо из Obsidian.

## Следующие шаги

1. **Запустите тестовый прогон** на 5 заметках
2. **Проверьте результаты** в логах и Anki
3. **При необходимости** исправьте заметки с ошибками
4. **Запустите полную синхронизацию** или инкрементальную
5. **Настройте регулярный запуск** (через cron или shell-скрипт)
6. **Изучите продвинутые возможности** в полной документации

---

## Дополнительные ресурсы

-   **Полная документация**: [README.md](README.md)
-   **Требования к проекту**: [.docs/REQUIREMENTS.md](.docs/REQUIREMENTS.md)
-   **APF формат**: [.docs/APF_FIELD_GUIDE.md](.docs/APF_FIELD_GUIDE.md)
-   **Настройка провайдеров**: [docs/LLM_PROVIDERS.md](docs/LLM_PROVIDERS.md)
-   **План интеграции агентов**: [.docs/AGENT_INTEGRATION_PLAN.md](.docs/AGENT_INTEGRATION_PLAN.md)

---

## Частые вопросы

### Можно ли обрабатывать только определённые папки?

Да, измените `source_dir` в `config.yaml`:

```yaml
source_dir: "40-Android" # Только Android заметки
```

### Можно ли выбрать другую колоду Anki?

Да, измените `anki_deck_name` в `config.yaml`:

```yaml
anki_deck_name: "Моя колода"
```

### Что делать, если карточка создана неправильно?

1. Удалите карточку в Anki
2. Исправьте заметку в Obsidian
3. Запустите синхронизацию снова:
    ```bash
    uv run obsidian-anki-sync sync --use-agents --config config.yaml
    ```

### Можно ли экспортировать без Anki?

Да! Используйте команду export:

```bash
uv run obsidian-anki-sync export --use-agents --config config.yaml
```

Это создаст файл `.apkg`, который можно импортировать в Anki на любом устройстве.

### Как работает инкрементальная синхронизация?

Инкрементальная синхронизация обрабатывает только заметки, которые:

-   Ещё не были синхронизированы
-   Были изменены после последней синхронизации (определяется по хешу содержимого)

Используйте `--incremental` для быстрой синхронизации новых заметок.

### Как проверить, какие заметки будут обработаны?

```bash
# Dry-run покажет все изменения без применения
uv run obsidian-anki-sync sync --dry-run --use-agents --config config.yaml

# Индекс покажет статистику
uv run obsidian-anki-sync index --config config.yaml
```

### Можно ли синхронизировать только определённую тему?

Да, измените `source_dir` в `config.yaml`:

```yaml
source_dir: "30-System-Design" # Только System Design
```

Или используйте символические ссылки для создания временной структуры.

### Как удалить все карточки из Anki?

1. Удалите базу данных: `rm .sync_state.db`
2. Удалите карточки вручную через Anki или используйте AnkiConnect API
3. Запустите синхронизацию заново

---

**Готово к использованию! Удачи с синхронизацией ваших заметок!**
