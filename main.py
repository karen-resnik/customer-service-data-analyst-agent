from src.agent import answer_question


def main() -> None:
    """Run an interactive CLI for the customer service data analyst agent."""
    print("Customer Service Data Analyst Agent")
    print("Ask questions about the Bitext customer service dataset.")
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
            answer = answer_question(user_input)
        except Exception as error:
            print(f"Error: {error}")
            continue

        print()
        print("Agent:")
        print(answer)
        print()


if __name__ == "__main__":
    main()