from src.evaluation.evaluate import evaluate_binary_counts


def main() -> None:
    result = evaluate_binary_counts(tp=0, fp=0, fn=0)
    print("Evaluation skeleton is working.")
    print(result)
    print("Next: wire this to the exact official Macro F0.5 definition.")


if __name__ == "__main__":
    main()
