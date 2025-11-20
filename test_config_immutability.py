#!/usr/bin/env python3
"""Test script to verify Config immutability."""

from pathlib import Path


def test_config_immutability():
    """Test that Config is immutable after initialization."""
    print("Testing Config immutability...")

    # Test 1: Try to modify a simple field
    print("\n1. Testing field modification on simple field (anki_deck_name)...")
    try:
        from src.obsidian_anki_sync.config import Config

        config = Config(
            vault_path=Path.cwd(),
            source_dir=Path("."),
            llm_provider="ollama",
        )
        print(f"   Initial deck name: {config.anki_deck_name}")
        config.anki_deck_name = "Different Deck"
        print("   ERROR: Field modification succeeded (should have failed)!")
        return False
    except Exception as e:
        print(f"   SUCCESS: Field modification blocked")
        print(f"   Error: {type(e).__name__}: {str(e)[:80]}")

    # Test 2: Try to modify a Path field
    print("\n2. Testing field modification on Path field (vault_path)...")
    try:
        from src.obsidian_anki_sync.config import Config

        config = Config(
            vault_path=Path.cwd(),
            source_dir=Path("."),
            llm_provider="ollama",
        )
        print(f"   Initial vault path: {config.vault_path}")
        config.vault_path = Path("/different/path")
        print("   ERROR: Path field modification succeeded (should have failed)!")
        return False
    except Exception as e:
        print(f"   SUCCESS: Path field modification blocked")
        print(f"   Error: {type(e).__name__}: {str(e)[:80]}")

    # Test 3: Try to modify using setattr
    print("\n3. Testing modification using setattr...")
    try:
        from src.obsidian_anki_sync.config import Config

        config = Config(
            vault_path=Path.cwd(),
            source_dir=Path("."),
            llm_provider="ollama",
        )
        setattr(config, "llm_temperature", 0.9)
        print("   ERROR: setattr modification succeeded (should have failed)!")
        return False
    except Exception as e:
        print(f"   SUCCESS: setattr modification blocked")
        print(f"   Error: {type(e).__name__}")

    # Test 4: Verify config can be created successfully
    print("\n4. Testing normal config creation...")
    try:
        from src.obsidian_anki_sync.config import Config

        config = Config(
            vault_path=Path.cwd(),
            source_dir=Path("."),
            llm_provider="ollama",
            llm_temperature=0.5,
        )
        print(f"   SUCCESS: Config created with temperature={config.llm_temperature}")
    except Exception as e:
        print(f"   ERROR: Config creation failed: {e}")
        return False

    # Test 5: Test reset_config function
    print("\n5. Testing reset_config() function...")
    try:
        from src.obsidian_anki_sync.config import (
            Config,
            get_config,
            reset_config,
            set_config,
        )

        # Set a config
        test_config = Config(
            vault_path=Path.cwd(),
            source_dir=Path("."),
            llm_provider="ollama",
        )
        set_config(test_config)

        # Get it back
        retrieved = get_config()
        print(f"   Config retrieved: {retrieved.llm_provider}")

        # Reset
        reset_config()
        print("   SUCCESS: reset_config() executed")

    except Exception as e:
        print(f"   ERROR: reset_config() failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("All immutability tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_config_immutability()
    exit(0 if success else 1)
