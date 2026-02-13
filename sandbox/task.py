import requests

def fetch_top_stories(limit: int = 5) -> list:
    url = "https://hacker-news.firebaseio.com/v0/topstories.json"
    response = requests.get(url)
    if response.status_code == 200:
        story_ids = response.json()[:limit]
        stories = []
        for story_id in story_ids:
            story_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
            story_response = requests.get(story_url)
            if story_response.status_code == 200:
                stories.append(story_response.json())
            else:
                print(f"Failed to fetch story {story_id}")
        return stories
    else:
        print("Failed to fetch top stories")
        return []