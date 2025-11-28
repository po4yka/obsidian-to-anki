import os


def print_separator(title):
    print("\n" + "=" * 80)
    print(f" {title} ")
    print("=" * 80 + "\n")


def check_content(file_path, keywords, name):
    try:
        with open(file_path) as f:
            content = f.read()

        missing = [k for k in keywords if k not in content]
        if missing:
            print(f"❌ {name}: Missing keywords: {missing}")
        else:
            print(f"✅ {name}: All keywords found.")
    except Exception as e:
        print(f"❌ {name}: Error reading file: {e}")


print("Verifying Prompts...")

base_path = "src/obsidian_anki_sync/agents"

# 1. improved_prompts.py
print_separator("improved_prompts.py")
check_content(
    os.path.join(base_path, "improved_prompts.py"),
    [
        "Cloze Deletions",
        "{{c1::",
        "MathJax",
        "\\(",
        "CardType: Cloze",
        "Special Features",
    ],
    "Improved Prompts",
)

# 2. card_splitting_prompts.py
print_separator("card_splitting_prompts.py")
check_content(
    os.path.join(base_path, "card_splitting_prompts.py"),
    ["Cloze Splitting", "{{c1::"],
    "Card Splitting Prompts",
)

# 3. context_enrichment_prompts.py
print_separator("context_enrichment_prompts.py")
check_content(
    os.path.join(base_path, "context_enrichment_prompts.py"),
    ["Math/Science Context", "SI Units", "math_science"],
    "Context Enrichment Prompts",
)

# 4. memorization_prompts.py
print_separator("memorization_prompts.py")
check_content(
    os.path.join(base_path, "memorization_prompts.py"),
    ["Cloze Effectiveness", "Bad Cloze"],
    "Memorization Prompts",
)

print("\nVerification Complete.")
