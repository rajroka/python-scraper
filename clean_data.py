import re
import json
import pandas as pd
from langdetect import detect


def is_english(text):
    try:
        return detect(text) == "en"
    except:
        return False


def clean_text(text):
    if not isinstance(text, str):
        return None
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def has_enough_words(text):
    english_words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
    total_words = text.split()
    if len(english_words) < 5:
        return False
    if len(total_words) > 0 and len(english_words) / len(total_words) < 0.6:
        return False
    return True


def process_instagram(input_file="instarawdata.json"):
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            insta_raw = json.load(f)
    except FileNotFoundError:
        print(f"{input_file} not found, skipping.")
        return []

    training_data = []

    for post in insta_raw:
        caption_raw = post.get("caption", "")
        hashtags_list = post.get("hashtags", [])

        if not caption_raw or not isinstance(caption_raw, str):
            continue
        if not is_english(caption_raw):
            continue
        if len(hashtags_list) < 3:
            continue

        clean = clean_text(caption_raw)
        if not clean or len(clean) < 30:
            continue
        if not has_enough_words(clean):
            continue
        if len(clean) > 400:
            clean = clean[:400]

        hashtags_str = " ".join([f"#{h}" for h in hashtags_list[:15]])
        training_data.append({
            "prompt": "Write a fitness and motivation Instagram caption with hashtags.",
            "completion": f"{clean}\n\n{hashtags_str}"
        })

    print(f"Instagram clean examples: {len(training_data)}")
    return training_data


def process_youtube(input_file="youtube_raw.csv"):
    try:
        youtube_df = pd.read_csv(input_file, on_bad_lines='skip', engine='python')
    except FileNotFoundError:
        print(f"{input_file} not found, skipping.")
        return []

    training_data = []

    for _, row in youtube_df.iterrows():
        description = row.get("description", "")
        title = row.get("title", "")

        if not isinstance(description, str):
            continue
        if not is_english(description):
            continue

        hashtags = re.findall(r'#\w+', description)
        if len(hashtags) < 3:
            continue

        clean = clean_text(description)
        if not clean or len(clean) < 30:
            continue
        if not has_enough_words(clean):
            continue
        if len(clean) > 400:
            clean = clean[:400]

        hashtags_str = " ".join(hashtags[:15])
        context = title if isinstance(title, str) and title.strip() else row.get('query', 'fitness motivation')

        training_data.append({
            "prompt": f"Write a fitness and motivation Instagram caption with hashtags for: {context}.",
            "completion": f"{clean}\n\n{hashtags_str}"
        })

    print(f"YouTube clean examples: {len(training_data)}")
    return training_data


def merge_and_save(output_file="training_data.csv"):
    training_data = []
    training_data.extend(process_instagram())
    training_data.extend(process_youtube())

    df = pd.DataFrame(training_data).drop_duplicates()
    df.to_csv(output_file, index=False)
    print(f"\nTotal training examples saved: {len(df)} → {output_file}")


if __name__ == "__main__":
    merge_and_save()