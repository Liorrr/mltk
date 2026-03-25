"""Interactive REPL for the chat engine."""

from __future__ import annotations

from mltk.chat.engine import ChatEngine


def chat_repl(results_path: str | None = None) -> None:
    """Start interactive chat session.

    Prints a welcome message, then loops reading stdin and printing responses.
    Type ``quit`` or ``exit`` to stop.

    Args:
        results_path: Optional path to a JSON results file produced by
            ``--mltk-export-json``. When omitted, the engine starts empty
            and still responds to ``help``, ``summary``, etc. with a prompt
            to load results.
    """
    engine = ChatEngine(results_path)

    print("mltk chat — ask questions about your test results")
    if results_path:
        total = len(engine.results)
        print(f"Loaded {total} result(s) from {results_path}")
    else:
        print("No results file loaded. Use --results-json to load one.")
    print("Type 'help' for available commands, 'quit' to exit")
    print()

    while True:
        try:
            question = input("mltk> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        answer = engine.ask(question)
        print(answer)
        print()
