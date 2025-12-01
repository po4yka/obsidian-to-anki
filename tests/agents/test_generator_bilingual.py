"""Tests for GeneratorAgent bilingual functionality."""

import json

import pytest

from obsidian_anki_sync.agents.generator import GeneratorAgent, ParsedCardStructure
from obsidian_anki_sync.models import Manifest, NoteMetadata, QAPair


@pytest.fixture()
def bilingual_generator_agent(mock_llm_provider):
    """Create a GeneratorAgent instance for bilingual testing."""
    return GeneratorAgent(
        ollama_client=mock_llm_provider, model="qwen3:32b", temperature=0.3
    )


@pytest.fixture()
def kotlin_singleton_metadata():
    """Metadata based on kotlin-224 note."""
    return NoteMetadata(
        id="kotlin-224",
        title="Prohibit Object Creation / Запрет создания объектов",
        topic="kotlin",
        language_tags=["en", "ru"],
        created="2025-10-15",
        updated="2025-11-10",
        subtopics=["classes", "design"],
        tags=[
            "classes",
            "constructor",
            "design",
            "difficulty/easy",
            "kotlin",
            "singleton",
        ],
        difficulty="easy",
    )


@pytest.fixture()
def kotlin_singleton_qa_pairs():
    """QA pairs from kotlin-224 note."""
    return [
        QAPair(
            card_index=1,
            question_en="How to programmatically prohibit creating a class object in Kotlin?",
            question_ru="Как программно запретить создание объекта (экземпляра) класса в Kotlin?",
            answer_en="""To prohibit external creation of class instances in Kotlin:

- Use a private constructor (primary or secondary) and expose only factory methods.
- Use an `object` declaration for singletons (no public constructor).

Other types like abstract or sealed classes cannot be instantiated directly, but they are modeling tools first, and should not be the primary answer to this question.

**Techniques:**
1. **Private primary constructor** – prevent direct instantiation and expose factories.
2. **Private secondary constructor** – when a non-private primary exists but some construction paths must be restricted.
3. **`object` declaration** – preferred for singletons; instance is created by the language, no `new`/constructor.
4. **Utility holder with private constructor** – to prevent instantiation when using companion/top-level-style utilities (though top-level functions or `object` are usually better in Kotlin).

### Code Examples

**Private primary constructor with factory (manual singleton-style access):**

```kotlin
class Singleton private constructor() {
    companion object {
        private var instance: Singleton? = null

        fun getInstance(): Singleton {
            return instance ?: synchronized(this) {
                instance ?: Singleton().also { instance = it }
            }
        }
    }

    fun doSomething() {
        println("Doing something")
    }
}

fun main() {
    // val singleton = Singleton()  // ERROR: Constructor is private

    val singleton = Singleton.getInstance()
    singleton.doSomething()

    val singleton2 = Singleton.getInstance()
    println(singleton === singleton2)  // true - same instance
}
```

Note: In Kotlin, prefer `object` declarations for singletons (see below) instead of manual patterns like this unless you specifically need lazy init with custom semantics.

**Object declaration (preferred for singletons):**

```kotlin
object DatabaseConnection {
    private var isConnected = false

    fun connect() {
        if (!isConnected) {
            println("Connecting to database...")
            isConnected = true
        }
    }

    fun disconnect() {
        if (isConnected) {
            println("Disconnecting from database...")
            isConnected = false
        }
    }

    fun isConnected() = isConnected
}

fun main() {
    // No constructor to call - object instance is managed by Kotlin

    DatabaseConnection.connect()
    println("Connected: ${DatabaseConnection.isConnected()}")

    DatabaseConnection.disconnect()
    println("Connected: ${DatabaseConnection.isConnected()}")
}
```""",
            answer_ru="""Чтобы программно запретить создание экземпляров класса извне в Kotlin, используют:

- Приватный конструктор (primary или secondary) плюс фабричные методы.
- `object`-декларацию для синглтонов (конструктора нет, экземпляр управляется языком).

Другие подходы (abstract/sealed классы) ограничивают создание экземпляров, но в первую очередь служат для моделирования и не являются базовым ответом на этот вопрос.

**Основные подходы:**
1. **Приватный primary constructor** – запрещаем прямое создание и открываем только контролируемые фабрики.
2. **Приватный secondary constructor** – если нужно ограничить отдельные пути создания.
3. **`object` declaration** – предпочтительный способ реализации синглтонов.
4. **Utility-класс с приватным конструктором** – чтобы нельзя было создать экземпляр, но в Kotlin обычно лучше использовать top-level функции или `object`.

### Примеры Кода

**Приватный primary constructor с фабрикой (ручной singleton-стиль):**

```kotlin
class Singleton private constructor() {
    companion object {
        private var instance: Singleton? = null

        fun getInstance(): Singleton {
            return instance ?: synchronized(this) {
                instance ?: Singleton().also { instance = it }
            }
        }
    }

    fun doSomething() {
        println("Doing something")
    }
}

fun main() {
    // val singleton = Singleton()  // ОШИБКА: конструктор приватный

    val singleton = Singleton.getInstance()
    singleton.doSomething()

    val singleton2 = Singleton.getInstance()
    println(singleton === singleton2)  // true - тот же экземпляр
}
```

Примечание: в Kotlin для синглтонов обычно следует использовать `object`, а не ручную реализацию, если нет особых требований к ленивой инициализации.

**Object declaration (предпочтительно для синглтонов):**

```kotlin
object DatabaseConnection {
    private var isConnected = false

    fun connect() {
        if (!isConnected) {
            println("Подключение к БД...")
            isConnected = true
        }
    }

    fun disconnect() {
        if (isConnected) {
            println("Отключение от БД...")
            isConnected = false
        }
    }

    fun isConnected() = isConnected
}

fun main() {
    // Нет конструктора для вызова — экземпляр управляется Kotlin

    DatabaseConnection.connect()
    println("Подключен: ${DatabaseConnection.isConnected()}")

    DatabaseConnection.disconnect()
    println("Подключен: ${DatabaseConnection.isConnected()}")
}
```

Примечание: Предпочтительнее object для синглтонов с контролируемой инициализацией.""",
        )
    ]


class TestBilingualGenerator:
    """Test bilingual card generation functionality."""

    def test_parse_card_structure(self, bilingual_generator_agent):
        """Test parsing APF HTML structure for translation."""
        # Sample APF HTML from English card
        apf_html = """<!-- Card 1 | slug: kotlin-prohibit-object-creation-1-en | CardType: Simple | Tags: kotlin classes design difficulty_easy -->

<!-- Title -->
How to programmatically prohibit creating a class object in Kotlin?

<!-- Key point (code block / image) -->
<pre><code class="language-kotlin">class Singleton private constructor() {
    companion object {
        private var instance: Singleton? = null

        fun getInstance(): Singleton {
            return instance ?: synchronized(this) {
                instance ?: Singleton().also { instance = it }
            }
        }
    }
}</code></pre>

<!-- Key point notes -->
<ul>
  <li>Use private constructor to prevent direct instantiation</li>
  <li>Expose factory methods for controlled creation</li>
  <li>Object declaration preferred for simple singletons</li>
</ul>

<!-- Other notes -->
<ul>
  <li>Ref: [[q-prohibit-object-creation--kotlin--easy.md#qa-1]]</li>
</ul>

<!-- manifest: {"slug":"kotlin-prohibit-object-creation-1-en","lang":"en","type":"Simple","tags":["kotlin","classes","design","difficulty_easy"]} -->"""

        result = bilingual_generator_agent._parse_card_structure(apf_html)

        assert (
            result.title
            == "How to programmatically prohibit creating a class object in Kotlin?"
        )
        assert result.key_point_code is not None
        assert "class Singleton private constructor()" in result.key_point_code
        assert len(result.key_point_notes) == 3
        assert "Use private constructor" in result.key_point_notes[0]
        assert len(result.other_notes) == 1
        assert "Ref:" in result.other_notes[0]

    def test_translation_prompt_generation(
        self, bilingual_generator_agent, kotlin_singleton_metadata
    ):
        """Test that translation prompts are generated correctly."""
        english_structure = ParsedCardStructure(
            title="How to programmatically prohibit creating a class object in Kotlin?",
            key_point_code='<pre><code class="language-kotlin">object Singleton { }</code></pre>',
            key_point_notes=[
                "Use private constructor to prevent direct instantiation",
                "Object declaration preferred for simple singletons",
                "Factory methods provide controlled access",
            ],
            other_notes=["Ref: [[test.md#qa-1]]"],
        )

        manifest = Manifest(
            slug="kotlin-prohibit-object-creation-1-ru",
            slug_base="kotlin-prohibit-object-creation",
            lang="ru",
            source_path="test.md",
            source_anchor="qa-1",
            note_id="kotlin-224",
            note_title="Prohibit Object Creation",
            card_index=1,
            guid="test-guid",
            hash6=None,
        )

        prompt = bilingual_generator_agent._build_translation_prompt(
            question="Как программно запретить создание объекта класса в Kotlin?",
            answer="Используйте приватный конструктор...",
            english_structure=english_structure,
            metadata=kotlin_singleton_metadata,
            manifest=manifest,
            lang="ru",
        )

        assert "Translate the TEXT CONTENT" in prompt
        assert "English Card Structure:" in prompt
        assert (
            json.dumps(english_structure.key_point_notes, ensure_ascii=False) in prompt
        )
        assert "PRESERVE the exact same card structure" in prompt
        assert "TRANSLATE ONLY the text content" in prompt

    def test_english_first_generation_order(
        self,
        bilingual_generator_agent,
        kotlin_singleton_metadata,
        kotlin_singleton_qa_pairs,
    ):
        """Test that English cards are generated first."""
        # Mock the LLM to return a simple APF response
        mock_response = """<!-- PROMPT_VERSION: apf-v2.1 -->
<!-- BEGIN_CARDS -->

<!-- Card 1 | slug: kotlin-prohibit-object-creation-1-en | CardType: Simple | Tags: kotlin classes design difficulty_easy -->

<!-- Title -->
How to prohibit object creation in Kotlin?

<!-- Key point (code block / image) -->
<pre><code class="language-kotlin">class Singleton private constructor() { }</code></pre>

<!-- Key point notes -->
<ul>
  <li>Use private constructor</li>
  <li>Object declaration preferred</li>
</ul>

<!-- manifest: {"slug":"kotlin-prohibit-object-creation-1-en","lang":"en","type":"Simple","tags":["kotlin","classes","design","difficulty_easy"]} -->

<!-- END_CARDS -->
END_OF_CARDS"""

        bilingual_generator_agent.ollama_client.set_response(
            "Test content", mock_response
        )

        # Generate cards
        result = bilingual_generator_agent.generate_cards(
            note_content="Test content",
            metadata=kotlin_singleton_metadata,
            qa_pairs=kotlin_singleton_qa_pairs,
            slug_base="kotlin-prohibit-object-creation",
        )

        # Verify English card was generated first
        assert len(result.cards) == 2  # Should have both EN and RU cards
        assert result.cards[0].lang == "en"
        assert result.cards[1].lang == "ru"

    def test_translation_assembly(
        self, bilingual_generator_agent, kotlin_singleton_metadata
    ):
        """Test that translated cards are assembled correctly from English structure."""
        english_structure = ParsedCardStructure(
            title="How to prohibit object creation in Kotlin?",
            key_point_code='<pre><code class="language-kotlin">object Singleton { }</code></pre>',
            key_point_notes=["Use private constructor", "Object declaration preferred"],
            other_notes=["Ref: [[test.md#qa-1]]"],
        )

        # Mock translated JSON response
        translated_json = """{
  "title": "Как запретить создание объектов в Kotlin?",
  "key_point_notes": ["Используйте приватный конструктор", "Object declaration предпочтителен"],
  "other_notes": ["Ref: [[test.md#qa-1]]"]
}"""

        manifest = Manifest(
            slug="kotlin-prohibit-object-creation-1-ru",
            slug_base="kotlin-prohibit-object-creation",
            lang="ru",
            source_path="test.md",
            source_anchor="qa-1",
            note_id="kotlin-224",
            note_title="Prohibit Object Creation",
            card_index=1,
            guid="test-guid",
            hash6=None,
        )

        result_html = bilingual_generator_agent._assemble_translated_card_html(
            english_structure=english_structure,
            translated_html=translated_json,
            metadata=kotlin_singleton_metadata,
            manifest=manifest,
        )

        assert "Как запретить создание объектов в Kotlin?" in result_html
        assert "Используйте приватный конструктор" in result_html
        assert "Object declaration предпочтителен" in result_html
        assert english_structure.key_point_code in result_html  # Code preserved
        assert "slug: kotlin-prohibit-object-creation-1-ru" in result_html
