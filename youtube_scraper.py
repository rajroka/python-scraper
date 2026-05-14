import requests
import pandas as pd
from config import API_KEY, SEARCH_QUERIES




def search_videos(query, max_results=50):
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "maxResults": max_results,
        "type": "video",
        "key": API_KEY
    }
    response = requests.get(url, params=params)
    return response.json().get("items", [])


def get_video_description(video_id):
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "snippet",
        "id": video_id,
        "key": API_KEY
    }
    response = requests.get(url, params=params)
    items = response.json().get("items", [])
    if items:
        return items[0]["snippet"]["description"]
    return ""


def scrape_youtube(output_file="youtube_raw.csv"):
    results = []

    for query in SEARCH_QUERIES:
        print(f"Searching: {query}")
        videos = search_videos(query)

        for video in videos:
            video_id = video["id"].get("videoId", "")
            if not video_id:
                continue
            description = get_video_description(video_id)
            if "#" in description and len(description) > 100:
                results.append({
                    "source": "youtube",
                    "query": query,
                    "title": video["snippet"]["title"],
                    "description": description
                })

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\nDone. Collected {len(df)} videos → {output_file}")


if __name__ == "__main__":
    scrape_youtube()