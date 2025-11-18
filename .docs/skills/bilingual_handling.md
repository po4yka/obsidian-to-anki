# Bilingual Handling Skill

**Purpose**: Handle bilingual content (EN/RU) in Obsidian notes and generate appropriate cards for each language.

**When to Use**: Load when processing notes that contain content in multiple languages.

---

## Core Principle: Language-Aware Processing

Both languages can exist in the same note. Generate cards that respect language boundaries while maintaining content relationships.

---

## Note Structure Recognition

### Bilingual Note Patterns

**Pattern 1: Parallel Sections**
```
# English Section
Q: What is X?
A: X is...

# Русская секция
В: Что такое X?
О: X это...
```

**Pattern 2: Interleaved Content**
```
English question:
Q: What is X?

Russian question:
В: Что такое X?

English answer:
A: X is...

Russian answer:
О: X это...
```

**Pattern 3: Metadata-Driven**
```yaml
---
title: "Concept Name"
lang: ["en", "ru"]
---
```

---

## Card Generation Strategy

### Option 1: Separate Cards per Language (Recommended)

**When**: Content is fully translated and independent.

**Approach**: Generate separate cards for each language.

**Example**:
```
English Card:
Q: What is a closure in JavaScript?
A: Function that retains access to outer scope variables
Tags: javascript closures english

Russian Card:
В: Что такое замыкание в JavaScript?
О: Функция, сохраняющая доступ к переменным внешней области видимости
Tags: javascript closures russian
```

**Benefits**:
- Each card is language-pure
- Learners can study in their preferred language
- No language mixing confusion

---

### Option 2: Bilingual Card

**When**: Content benefits from seeing both languages together.

**Approach**: Include both languages in one card.

**Example**:
```
Q: What is a closure in JavaScript? / Что такое замыкание в JavaScript?
A: Function that retains access to outer scope variables / Функция, сохраняющая доступ к переменным внешней области видимости
Tags: javascript closures bilingual
```

**Benefits**:
- Helps language learning
- Reinforces understanding through translation
- Useful for technical vocabulary

**Use Cases**:
- Language learning contexts
- Technical terminology that needs translation
- When both languages add value

---

## Language Detection

### Detection Methods

1. **Metadata**: Check YAML frontmatter for `lang` field
2. **Content Analysis**: Detect Cyrillic vs. Latin characters
3. **Section Headers**: Look for language markers ("English", "Русский", etc.)
4. **Pattern Matching**: Identify Q/A patterns in each language

### Detection Rules

**English Indicators**:
- Latin alphabet
- Question words: "What", "How", "Why", "When"
- Question mark: "?"

**Russian Indicators**:
- Cyrillic alphabet
- Question words: "Что", "Как", "Почему", "Когда"
- Question mark: "?"

---

## Tagging Strategy

### Language Tags

Always include language tag:
- `english` or `en` for English content
- `russian` or `ru` for Russian content
- `bilingual` for cards with both languages

### Tag Order

Follow standard tag order with language tag:
```
<language/tool> <platform> <domain> <language_tag>
```

**Examples**:
- `javascript closures english`
- `kotlin coroutines russian`
- `python algorithms bilingual`

---

## Content Extraction

### Extracting Q/A Pairs

**For Parallel Sections**:
1. Extract English Q/A pairs from English section
2. Extract Russian Q/A pairs from Russian section
3. Match pairs by position or content similarity
4. Generate cards accordingly

**For Interleaved Content**:
1. Identify language of each Q/A block
2. Group by language
3. Generate cards per language group

**For Metadata-Driven**:
1. Check `lang` field in frontmatter
2. Extract content for each specified language
3. Generate cards per language

---

## Quality Checks

### Language Purity

- ✓ English cards contain only English content
- ✓ Russian cards contain only Russian content
- ✓ Bilingual cards clearly separate languages
- ✓ No language mixing within single card (unless intentional)

### Content Consistency

- ✓ Q/A pairs match across languages (if generating parallel cards)
- ✓ Technical terms are consistent
- ✓ Examples work in both languages (if applicable)

### Tagging Accuracy

- ✓ Language tag matches card content
- ✓ Tags follow taxonomy rules
- ✓ No orphan language tags

---

## Common Patterns

### Pattern 1: Technical Interview Prep

**Structure**: English question, Russian explanation (or vice versa)

**Approach**: Generate bilingual card to help with technical vocabulary.

**Example**:
```
Q: What is time complexity? / Что такое временная сложность?
A: Measure of algorithm efficiency / Мера эффективности алгоритма
Tags: algorithms complexity bilingual
```

---

### Pattern 2: Language Learning

**Structure**: Concept in one language, translation in other

**Approach**: Generate bilingual card for vocabulary building.

**Example**:
```
Q: What does "closure" mean in programming? / Что означает "замыкание" в программировании?
A: Function retaining outer scope access / Функция, сохраняющая доступ к внешней области
Tags: programming concepts bilingual
```

---

### Pattern 3: Independent Content

**Structure**: Completely separate content per language

**Approach**: Generate separate cards per language.

**Example**:
```
English Card:
Q: What is React?
A: JavaScript library for building UIs
Tags: javascript react english

Russian Card:
В: Что такое React?
О: JavaScript библиотека для создания UI
Tags: javascript react russian
```

---

## Edge Cases

### Mixed Language Content

**Problem**: Some content mixes languages (e.g., English code, Russian comments).

**Solution**:
- Keep code in original language
- Translate comments/explanations
- Tag appropriately (usually `bilingual`)

### Incomplete Translations

**Problem**: Note has English but missing Russian (or vice versa).

**Solution**:
- Generate cards only for available language
- Tag with available language only
- Don't create empty or placeholder cards

### Language-Specific Concepts

**Problem**: Concept exists only in one language (e.g., language-specific syntax).

**Solution**:
- Generate card in that language only
- Add note explaining language specificity if needed
- Tag with appropriate language tag

---

## Best Practices

✓ **Respect language boundaries**: Don't mix languages unintentionally
✓ **Consistent tagging**: Always include language tag
✓ **Match content**: Ensure Q/A pairs align across languages
✓ **Quality over quantity**: Better to have one good card per language than poor bilingual cards
✓ **User preference**: Consider generating both options and letting user choose

---

## Common Mistakes to Avoid

❌ **Language mixing**: Mixing English and Russian inappropriately
❌ **Missing language tags**: Forgetting to tag cards with language
❌ **Inconsistent extraction**: Extracting Q/A pairs incorrectly
❌ **Forced bilingualism**: Creating bilingual cards when separate cards are better
❌ **Ignoring metadata**: Not checking frontmatter for language hints

