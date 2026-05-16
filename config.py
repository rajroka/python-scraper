# Configuration for YouTube Scraper

import os


def get_api_key() -> str:
    """Return the YouTube Data API key from the YOUTUBE_API_KEY environment variable.

    Raises:
        EnvironmentError: If the variable is unset, empty, or whitespace-only.
    """
    value = os.environ.get("YOUTUBE_API_KEY", "")
    stripped = value.strip()
    if not stripped:
        if value:
            raise EnvironmentError(
                "YOUTUBE_API_KEY is set but contains only whitespace. "
                "Please set it to a valid API key: "
                "export YOUTUBE_API_KEY=your-api-key-here"
            )
        raise EnvironmentError(
            "YOUTUBE_API_KEY environment variable is not set or is empty. "
            "Please set it before running the scraper: "
            "export YOUTUBE_API_KEY=your-api-key-here"
        )
    return value

# Search queries to scrape videos for
SEARCH_QUERIES = [
    "fitness motivation gym workout",
    "gym motivation no excuses",
    "daily gym motivation video",
    "intense workout motivation gym",
    "bodybuilding motivation workout",
    "beast mode gym motivation",
    "gym discipline motivation speech",
    "fitness transformation motivation",
    "gym grind motivation video",
    "workout consistency motivation",
    "early morning gym motivation",
    "strength training motivation",
    "push day motivation gym",
    "leg day motivation gym",
    "chest workout motivation gym",
    "back workout motivation gym",
    "arm workout motivation gym",
    "full body workout motivation",
    "home workout motivation no equipment",
    "calisthenics workout motivation",
    "fat loss workout motivation",
    "weight loss gym motivation",
    "fitness journey transformation",

    "self discipline motivation speech",
    "discipline equals freedom motivation",
    "focus and discipline motivation",
    "stop wasting time motivation speech",
    "hard work discipline motivation",
    "build habits motivation video",
    "consistency motivation speech",
    "never give up motivation video",
    "mental discipline motivation",
    "stay focused motivation speech",
    "no excuses discipline motivation",
    "daily routine discipline motivation",
    "productivity motivation speech",
    "life discipline motivation",
    "success discipline mindset",

    "success motivation speech",
    "hustle grind motivation video",
    "millionaire mindset motivation",
    "entrepreneur motivation speech",
    "business success motivation",
    "financial freedom motivation speech",
    "success habits motivation video",
    "grind mindset motivation",
    "work hard success motivation",
    "rich mindset motivation",
    "startup motivation speech",
    "motivational success story",
    "never settle motivation speech",
    "goals achievement motivation",
    "success journey motivation",

    "strong mindset motivation video",
    "mental strength motivation speech",
    "build confidence motivation video",
    "positive mindset motivation speech",
    "overcome fear motivation video",
    "self belief motivation speech",
    "mental toughness motivation",
    "mindset shift motivation video",
    "think positive motivation speech",
    "focus mindset motivation video",
    "emotional strength motivation",
    "mental resilience motivation speech",

    "life motivation speech video",
    "inspirational motivation video",
    "motivational quotes speech video",
    "morning motivation video",
    "wake up motivation speech",
    "change your life motivation video",
    "powerful motivational speech",
    "study motivation video",
    "exam motivation speech",
    "future goals motivation video",
    "life changing motivation speech",
    "daily motivation video",

    "fitness lifestyle motivation",
    "healthy lifestyle motivation gym",
    "body transformation motivation video",
    "fitness journey story motivation",
    "gym lifestyle motivation video",
    "healthy habits motivation video",
    "transformation before after fitness",
    "natural body transformation motivation",
    "fitness influencer motivation",
    "workout lifestyle motivation vlog",

    "gym motivation shorts",
    "fitness motivation youtube shorts",
    "viral gym motivation clips",
    "short motivational video gym",
    "30 second motivation speech",
    "instagram reels fitness motivation",
    "youtube shorts discipline motivation",
    "viral motivational speech clips",
    "aesthetic gym motivation edit"
]
