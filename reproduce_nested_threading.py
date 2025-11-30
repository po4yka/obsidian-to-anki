import concurrent.futures
import threading
import time


def simulate_card_generation(duration=0.1):
    time.sleep(duration)
    return 1


def process_note_nested(num_cards, max_workers):
    # Simulate inner thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(simulate_card_generation) for _ in range(num_cards)]
        return sum(f.result() for f in futures)


def process_note_sequential(num_cards):
    # Simulate sequential processing
    return sum(simulate_card_generation() for _ in range(num_cards))


def run_nested_test(num_notes, cards_per_note, max_workers):
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_note_nested, cards_per_note, max_workers)
            for _ in range(num_notes)
        ]
        results = sum(f.result() for f in futures)
    return time.time() - start


def run_flat_test(num_notes, cards_per_note, max_workers):
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_note_sequential, cards_per_note)
            for _ in range(num_notes)
        ]
        results = sum(f.result() for f in futures)
    return time.time() - start


if __name__ == "__main__":
    NUM_NOTES = 20
    CARDS_PER_NOTE = 5
    MAX_WORKERS = 5

    print(
        f"Simulating {NUM_NOTES} notes, {CARDS_PER_NOTE} cards/note, max_workers={MAX_WORKERS}"
    )

    nested_time = run_nested_test(NUM_NOTES, CARDS_PER_NOTE, MAX_WORKERS)
    print(f"Nested ThreadPools Time: {nested_time:.4f}s")

    flat_time = run_flat_test(NUM_NOTES, CARDS_PER_NOTE, MAX_WORKERS)
    print(f"Flat (Outer Parallel, Inner Sequential) Time: {flat_time:.4f}s")

    print(f"Speedup: {nested_time / flat_time:.2f}x")
