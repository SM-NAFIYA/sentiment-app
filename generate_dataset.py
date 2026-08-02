"""
Generates a larger, more realistic labeled sentiment dataset
(data/sentiment_data.csv) -- Positive / Negative / Neutral.

Unlike the earlier version, this one mixes in casual, lowercase,
contraction-heavy phrasing ("i love this", "not worth it", "its ok i
guess") alongside more formal review sentences, so the trained model
generalizes to everyday typed text instead of only recognizing rigid
templates.
"""
import csv
import os
import random
os.makedirs("data",exit_ok=True)
random.seed(7)

subjects = [
    "this product", "this laptop", "this phone", "the food", "the app",
    "this book", "the hotel", "the flight", "the movie", "the game",
    "this restaurant", "the course", "the service", "the delivery",
    "the software", "the staff", "the event", "this item", "the price",
    "the quality", "it",
]

# --- Casual, everyday phrasing (short, lowercase, contractions) -----------
positive_casual = [
    "i love {s}", "i really like {s}", "i like {s} a lot",
    "{s} is great", "{s} is awesome", "{s} is amazing",
    "{s} is so good", "loving {s} so far", "{s} exceeded my expectations",
    "can't stop using {s}", "{s} works perfectly", "{s} is exactly what i needed",
    "highly recommend {s}", "{s} made my day", "{s} is fantastic honestly",
    "really happy with {s}", "{s} is worth every penny", "obsessed with {s}",
    "{s} is better than expected", "no complaints about {s} at all",
]

negative_casual = [
    "i hate {s}", "i really dislike {s}", "i don't like {s}",
    "{s} is terrible", "{s} is awful", "{s} is bad",
    "{s} is so disappointing", "not happy with {s}", "{s} broke after a week",
    "waste of money on {s}", "{s} doesn't work properly", "{s} is not worth it",
    "wouldn't recommend {s}", "{s} ruined my day", "{s} is garbage honestly",
    "really unhappy with {s}", "{s} is a rip off", "regret buying {s}",
    "{s} is worse than expected", "so many problems with {s}",
]

neutral_casual = [
    "{s} is okay", "{s} is fine i guess", "{s} is average",
    "not sure how i feel about {s}", "{s} is alright, nothing special",
    "{s} does the job", "{s} arrived on time", "{s} is as described",
    "{s} is similar to other options", "still deciding what i think about {s}",
    "{s} has pros and cons", "{s} is what you'd expect",
    "{s} is neither good nor bad", "mixed feelings about {s}",
    "{s} was okay overall", "it's a normal {s}, nothing more",
]

# --- More formal, review-style phrasing (kept from before, smaller set) ---
positive_formal = [
    "I thought {s} was excellent.",
    "{S} was absolutely wonderful, I loved it.",
    "Really impressive experience with {s}.",
    "Honestly, {s} exceeded my expectations completely.",
    "{S} is outstanding and I'd highly recommend it.",
]

negative_formal = [
    "I thought {s} was terrible.",
    "{S} was absolutely awful, I hated it.",
    "Really disappointing experience with {s}.",
    "Honestly, {s} fell far short of my expectations.",
    "{S} is dreadful and I would not recommend it.",
]

neutral_formal = [
    "{S} was okay, nothing special.",
    "{S} is fine, I have no strong opinion.",
    "The price of {s} is about average.",
    "{S} has both good and bad points.",
    "{S} works as described in the manual.",
]

# --- Generic phrases not tied to a specific subject, wider vocabulary -----
positive_generic = [
    "best purchase ever", "best experience of my life", "this made my whole week",
    "couldn't be happier with this", "top notch quality", "exceeded all my expectations",
    "genuinely impressed", "such a pleasant surprise", "flawless from start to finish",
    "would buy again in a heartbeat", "outstanding value for money",
    "this is exactly what i wanted", "brilliant, just brilliant",
    "everything about this was great", "superb experience overall",
    "delighted with how this turned out", "phenomenal, no other word for it",
]

negative_generic = [
    "worst purchase ever", "worst experience of my life", "this ruined my whole week",
    "couldn't be more disappointed", "poor quality overall", "fell short of all my expectations",
    "genuinely frustrated", "such an unpleasant surprise", "a disaster from start to finish",
    "would never buy again", "terrible value for money",
    "this is not what i wanted at all", "horrible, just horrible",
    "everything about this was bad", "awful experience overall",
    "regret how this turned out", "dreadful, no other word for it",
]

neutral_generic = [
    "it's fine, nothing to write home about", "an average experience overall",
    "neither impressed nor disappointed", "does what it says, nothing more",
    "middle of the road, honestly", "some good, some bad, evens out",
    "not the best but not the worst either", "an unremarkable experience",
    "hard to say if it's good or bad", "just an ordinary experience",
]


def fill(template, subj):
    return template.format(s=subj, S=subj.capitalize())


rows = []
for subj in subjects:
    for t in positive_casual:
        rows.append((fill(t, subj), "positive"))
    for t in negative_casual:
        rows.append((fill(t, subj), "negative"))
    for t in neutral_casual:
        rows.append((fill(t, subj), "neutral"))
    for t in positive_formal:
        rows.append((fill(t, subj), "positive"))
    for t in negative_formal:
        rows.append((fill(t, subj), "negative"))
    for t in neutral_formal:
        rows.append((fill(t, subj), "neutral"))

# Generic phrases are added once each (not per-subject) since they stand alone
for t in positive_generic:
    rows.append((t, "positive"))
for t in negative_generic:
    rows.append((t, "negative"))
for t in neutral_generic:
    rows.append((t, "neutral"))

# Repeat generic phrases a few times so they carry enough weight in training
for _ in range(15):
    for t in positive_generic:
        rows.append((t, "positive"))
    for t in negative_generic:
        rows.append((t, "negative"))
    for t in neutral_generic:
        rows.append((t, "neutral"))

random.shuffle(rows)

# Balance classes so none dominates
from collections import defaultdict
by_label = defaultdict(list)
for text, label in rows:
    by_label[label].append((text, label))

min_count = min(len(v) for v in by_label.values())
balanced_rows = []
for label, items in by_label.items():
    balanced_rows.extend(items[:min_count])
random.shuffle(balanced_rows)

with open("data/sentiment_data.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "label"])
    writer.writerows(balanced_rows)

print(f"Wrote {len(balanced_rows)} rows to data/sentiment_data.csv")
