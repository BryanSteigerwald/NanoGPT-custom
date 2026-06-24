# csv_to_txt.py
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, required=True)
parser.add_argument("--out", type=str, default="data/raw/converted.txt")
args = parser.parse_args()

with open(args.csv, "r", encoding="utf-8", errors="ignore") as f_in, \
     open(args.out, "w", encoding="utf-8") as f_out:
    reader = csv.DictReader(f_in)
    for row in reader:
        name        = row.get("Name", "").strip()
        about       = row.get("About the game", "").strip()
        genres      = row.get("Genres", "").strip()
        tags        = row.get("Tags", "").strip()
        reviews     = row.get("Reviews", "").strip()
        release     = row.get("Release date", "").strip()
        playtime    = row.get("Average playtime forever", "").strip()

        if not about:
            continue  # skip games with no description

        doc = (
            f"Game: {name}\n"
            f"Release Date: {release}\n"
            f"Genres: {genres}\n"
            f"Tags: {tags}\n"
            f"Average Playtime: {playtime} minutes\n"
            f"\n{about}"
        )
        if reviews:
            doc += f"\n\nReviews: {reviews}"

        f_out.write(doc + "\n\n")

print(f"Done. Written to {args.out}")