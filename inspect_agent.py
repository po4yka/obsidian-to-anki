import inspect

from pydantic_ai import Agent


def main():
    print("Inspecting Agent...")
    print(f"Agent bases: {Agent.__bases__}")

    sig = inspect.signature(Agent.__init__)
    print(f"Agent.__init__ signature: {sig}")

    run_sig = inspect.signature(Agent.run)
    print(f"Agent.run signature: {run_sig}")

    # Check if AbstractAgent is in bases
    for base in Agent.__mro__:
        print(f"Base: {base.__name__}")


if __name__ == "__main__":
    main()
