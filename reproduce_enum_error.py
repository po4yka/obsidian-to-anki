
from obsidian_anki_sync.agents.models import CardSplittingResult
from pydantic import ValidationError

def main():
    print("Checking CardSplittingResult schema...")
    schema = CardSplittingResult.model_json_schema()
    properties = schema.get('properties', {})
    strategy = properties.get('splitting_strategy', {})

    print(f"Type: {strategy.get('type')}")
    print(f"Enum: {strategy.get('enum')}")

    try:
        print("\nTrying to validate 'prerequisite_aware'...")
        CardSplittingResult(
            should_split=True,
            card_count=2,
            splitting_strategy="prerequisite_aware",
            split_plan=[],
            reasoning="test",
            decision_time=0.1,
            confidence=0.9
        )
        print("Validation SUCCESS")
    except ValidationError as e:
        print(f"Validation FAILED: {e}")

if __name__ == "__main__":
    main()
