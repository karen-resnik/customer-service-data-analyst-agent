import argparse

from src.agent import answer_question_with_trace

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Customer Service Data Analyst Agent CLI"
    )
    parser.add_argument(
        "--session",
        default="default",
        help="Session ID used for persistent conversation memory.",
    )
    return parser.parse_args()

def format_error_message(error: Exception) -> str:
    """Convert internal exceptions into user-friendly CLI messages."""
    error_text = str(error)

    if "NEBIUS_API_KEY is not set" in error_text:
        return (
            "The Nebius API key is missing. Please add NEBIUS_API_KEY to your .env file "
            "and try again."
        )

    if "Unknown category" in error_text:
        return (
            "I could not match that to a known dataset category. "
            "Try using one of the dataset categories, such as ACCOUNT, REFUND, ORDER, "
            "PAYMENT, SHIPPING, or CONTACT."
        )

    if "Unknown intent" in error_text:
        return (
            "I could not match that to a known dataset intent. "
            "Try asking in broader natural language, or ask me to list the available intents."
        )

    if "recursion_limit" in error_text or "need more steps" in error_text.lower():
        return (
            "I needed more reasoning steps than the current limit allows. "
            "Try asking a narrower question."
        )

    return (
        "Something went wrong while answering your question. "
        "Please try rephrasing it or asking a narrower dataset question."
    )


def main() -> None:
    """Run an interactive CLI for the customer service data analyst agent."""
    args = parse_args()
    session_id = args.session
    
    print("Customer Service Data Analyst Agent")
    print("Ask questions about the Bitext customer service dataset.")
    print(f"Session: {session_id}")
    print("Type 'exit' or 'quit' to stop.")
    print()

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not user_input:
            continue

        try:
            answer, trace_steps = answer_question_with_trace(user_input, session_id=session_id,)
        except Exception as error:
            print()
            print("Agent:")
            print(format_error_message(error))
            print()
            continue

        print()
        print("Reasoning / tool trace:")
        for step_number, step in enumerate(trace_steps, start=1):
            print(f"\n[{step_number}] {step}")

        print()
        print("Agent:")
        print(answer)
        print()


if __name__ == "__main__":
    main()