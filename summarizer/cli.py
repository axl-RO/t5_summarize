import argparse
from summarizer.model import summarize_batch


def main():
    parser = argparse.ArgumentParser(description="FLAN-T5 Text Summarizer")

    parser.add_argument("--text", type=str, help="Text to summarize")
    parser.add_argument("--file", type=str, help="File containing text")
    parser.add_argument("--style", choices=["short", "medium"], default="medium")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--beams", type=int, default=4)
    parser.add_argument("--batch", type=int, default=1)

    args = parser.parse_args()

    texts = []

    if args.text:
        texts.append(args.text)

    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            raw = f.read()

        if args.batch > 1:
            chunks = raw.split("\n\n")
            texts = chunks[:args.batch]
        else:
            texts = [raw]

    else:
        print("Provide --text or --file")
        return

    summaries = summarize_batch(
        texts,
        style=args.style,
        temperature=args.temperature,
        num_beams=args.beams
    )

    for i, s in enumerate(summaries, 1):
        print(f"\nSummary {i}:\n{s}\n")


if __name__ == "__main__":
    main()
