import os
import googleapiclient.discovery
import logging
import json
import re
import time
import ollama
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ── Configuration ────────────────────────────────────────────────────────────
MAX_COMMENTS   = 500  # Maximum comments to fetch per run
CHUNK_SIZE     = 50    # Comments per API call
RETRY_ATTEMPTS = 3     # Number of retries on failure
RETRY_DELAY    = 5     # Seconds to wait between retries
OLLAMA_MODEL   = "mistral"  # Model to use — change to "llama3.2" if needed
# ─────────────────────────────────────────────────────────────────────────────


class YouTubeCommentsFetcher:
    """Fetches top comments from a YouTube video using the YouTube Data API."""

    def __init__(self):
        self.api_key = os.getenv("YOUTUBE_API_KEY")
        if not self.api_key:
            raise ValueError("YOUTUBE_API_KEY not found in .env file.")
        self.youtube = googleapiclient.discovery.build(
            "youtube", "v3", developerKey=self.api_key
        )

    def extract_video_id(self, url: str):
        try:
            match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
            return match.group(1) if match else None
        except Exception as e:
            logging.error(f"Video ID extraction error: {e}")
            return None

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\\u[0-9A-Fa-f]{4}', '', text)
        text = re.sub(r'\\U[0-9A-Fa-f]{8}', '', text)
        text = re.sub(r'[\\\n\t\r]', ' ', text)
        text = re.sub(r'"+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def fetch(self, video_url: str) -> list:
        video_id = self.extract_video_id(video_url)
        if not video_id:
            logging.error("Invalid YouTube URL.")
            return []

        comments = []
        next_page_token = None
        logging.info(f"Fetching up to {MAX_COMMENTS} comments...")

        with tqdm(total=MAX_COMMENTS, desc="📥 Fetching comments", unit="comment") as pbar:
            while len(comments) < MAX_COMMENTS:
                try:
                    request = self.youtube.commentThreads().list(
                        part="snippet",
                        videoId=video_id,
                        pageToken=next_page_token,
                        maxResults=min(100, MAX_COMMENTS - len(comments)),
                        order="relevance",
                        textFormat="plainText"
                    )
                    response = request.execute()

                    for item in response["items"]:
                        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                        comment = self.clean_text(comment)
                        if comment:
                            comments.append(comment)
                            pbar.update(1)

                    next_page_token = response.get("nextPageToken")
                    if not next_page_token:
                        break

                except Exception as e:
                    logging.error(f"Error fetching comments: {e}")
                    break

        logging.info(f"✅ Fetched {len(comments)} comments.")
        self._save(comments, "comments.json")
        return comments

    def _save(self, comments: list, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(comments, f, ensure_ascii=False, indent=4)
        logging.info(f"Comments saved to {path}")


class PoliticalCommentAnalyser:
    """
    Analyses YouTube comments for political leaning and sentiment using Ollama (local).

    Political leaning score:
        -1.0  ->  Strongly Democrat / Left-leaning
         0.0  ->  Neutral / Non-political
        +1.0  ->  Strongly Republican / Right-leaning

    Categories:
        - Supportive
        - Critical
        - Republican-leaning
        - Democrat-leaning
        - Neutral/Informational
        - Emotional/Reactive
        - Spam/Irrelevant
    """

    CATEGORIES = [
        "Supportive",
        "Critical",
        "Republican-leaning",
        "Democrat-leaning",
        "Neutral/Informational",
        "Emotional/Reactive",
        "Spam/Irrelevant"
    ]

    def __init__(self):
        # Verify Ollama is running and model is available
        try:
            models = ollama.list()
            available = [m.model for m in models.models]
            if not any(OLLAMA_MODEL in m for m in available):
                raise ValueError(f"Model '{OLLAMA_MODEL}' not found. Run: ollama pull {OLLAMA_MODEL}")
            logging.info(f"✅ Ollama model '{OLLAMA_MODEL}' is ready.")
        except Exception as e:
            raise RuntimeError(f"Ollama error: {e}. Make sure Ollama is running.")

    def _build_prompt(self, comments: list) -> str:
        numbered = "\n".join([f"{i+1}. {c}" for i, c in enumerate(comments)])
        categories_str = ", ".join(self.CATEGORIES)
        return f"""You are an expert political analyst specialising in US political discourse and media analysis.
Analyse the following YouTube comments from a political debate/news video.

For EACH comment, return a JSON object with these exact fields:
- "comment": the original comment text (copy exactly)
- "category": one of [{categories_str}]
- "political_score": a float from -1.0 (strongly Democrat/Left) to +1.0 (strongly Republican/Right), 0.0 = neutral
- "sentiment": one of ["positive", "negative", "neutral"]
- "confidence": a float from 0.0 to 1.0 indicating how confident you are in your classification
- "sarcasm_detected": true or false
- "reasoning": one sentence explaining your classification

Return ONLY a valid JSON array of objects. No extra text, no markdown, no code blocks, no trailing commas.

CRITICAL RULES:
1. SARCASM: If a comment is sarcastic, FLIP the political_score. E.g. "Oh yeah Trump is SO great" is actually Democrat-leaning, score should be negative. Set sarcasm_detected to true.
2. CONTEXT AWARENESS: Comments like "he's right" or "I agree" must be interpreted based on WHO they are agreeing with. If agreeing with a Republican figure, score positive. If agreeing with a Democrat figure, score negative.
3. CODED LANGUAGE: Detect dog whistles and coded political language. E.g. "globalists", "deep state", "stolen election" = Republican-leaning. "systemic racism", "equity", "mostly peaceful" = Democrat-leaning.
4. CONFIDENCE: Ambiguous or very short comments should have low confidence (0.0-0.4). Clear political statements should have high confidence (0.7-1.0).

SCORING GUIDE:
  -1.0 to -0.6 : Strongly Democrat / Progressive / Left-wing
  -0.6 to -0.2 : Mildly Democrat / Centre-left
  -0.2 to +0.2 : Neutral / Non-political / Centrist
  +0.2 to +0.6 : Mildly Republican / Centre-right
  +0.6 to +1.0 : Strongly Republican / Conservative / Right-wing

CATEGORY GUIDE:
  Supportive            - Agrees with or praises the video/topic/person
  Critical              - Disagrees with or criticises the video/topic/person
  Republican-leaning    - Uses conservative talking points (pro-gun, anti-immigration, low taxes, Trump support, election fraud claims)
  Democrat-leaning      - Uses liberal talking points (pro-choice, climate change, social justice, Biden/Harris support, anti-Trump)
  Neutral/Informational - Shares facts, asks questions, no clear bias
  Emotional/Reactive    - Strong emotion (anger, excitement) with no clear political stance
  Spam/Irrelevant       - Off-topic, promotional, non-English, bot-like, or meaningless

COMMENTS TO ANALYSE:
{numbered}"""

    def _call_ollama(self, prompt: str, attempt: int = 1):
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0}
            )
            return response["message"]["content"]
        except Exception as e:
            if attempt < RETRY_ATTEMPTS:
                logging.warning(f"Ollama error (attempt {attempt}): {e}. Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
                return self._call_ollama(prompt, attempt + 1)
            logging.error(f"Ollama failed after {RETRY_ATTEMPTS} attempts: {e}")
            return None

    def _parse_response(self, raw: str) -> list:
        try:
            # Strip any accidental markdown code fences
            cleaned = re.sub(r"```(?:json)?", "", raw).strip()
            cleaned = cleaned.rstrip("`").strip()
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logging.error(f"JSON parse error: {e}")
            logging.error(f"Raw response: {raw[:500]}")
            return []

    def _init_unified(self) -> dict:
        """Initialises the single unified results dictionary."""
        return {
            "total_comments_analysed": 0,
            "overall_political_score": 0.0,
            "overall_leaning": "",
            "sarcasm_detected_count": 0,
            "sentiment_breakdown": {
                "positive": 0,
                "negative": 0,
                "neutral":  0
            },
            "category_breakdown": {cat: [] for cat in self.CATEGORIES},
            "score_distribution": {
                "strongly_democrat":   0,
                "mildly_democrat":     0,
                "neutral":             0,
                "mildly_republican":   0,
                "strongly_republican": 0
            },
            "_all_scores": [],  # Internal — removed before saving
            "raw_results": []
        }

    def _append_to_unified(self, results: list, unified: dict):
        """Appends each comment from a chunk directly into the correct section of the unified dict."""
        for item in results:
            unified["raw_results"].append(item)
            unified["total_comments_analysed"] += 1

            # ── Sarcasm count ─────────────────────────────────────────────────
            if item.get("sarcasm_detected", False):
                unified["sarcasm_detected_count"] += 1

            # ── Category ──────────────────────────────────────────────────────
            cat = item.get("category", "Neutral/Informational")
            if cat not in unified["category_breakdown"]:
                cat = "Neutral/Informational"
            unified["category_breakdown"][cat].append({
                "comment":          item.get("comment", ""),
                "political_score":  item.get("political_score", 0.0),
                "sentiment":        item.get("sentiment", "neutral"),
                "confidence":       item.get("confidence", 0.0),
                "sarcasm_detected": item.get("sarcasm_detected", False),
                "reasoning":        item.get("reasoning", "")
            })

            # ── Sentiment ─────────────────────────────────────────────────────
            sentiment = item.get("sentiment", "neutral")
            if sentiment in unified["sentiment_breakdown"]:
                unified["sentiment_breakdown"][sentiment] += 1

            # ── Score distribution ────────────────────────────────────────────
            score = float(item.get("political_score", 0.0))
            unified["_all_scores"].append(score)

            if score <= -0.6:
                unified["score_distribution"]["strongly_democrat"] += 1
            elif score <= -0.2:
                unified["score_distribution"]["mildly_democrat"] += 1
            elif score < 0.2:
                unified["score_distribution"]["neutral"] += 1
            elif score < 0.6:
                unified["score_distribution"]["mildly_republican"] += 1
            else:
                unified["score_distribution"]["strongly_republican"] += 1

    def _finalise(self, unified: dict):
        """Calculates the final overall score and leaning label, removes internal keys."""
        scores = unified.pop("_all_scores", [])
        avg    = sum(scores) / len(scores) if scores else 0.0
        unified["overall_political_score"] = round(avg, 4)

        if avg <= -0.6:
            unified["overall_leaning"] = "Strongly Democrat / Left-leaning"
        elif avg <= -0.2:
            unified["overall_leaning"] = "Mildly Democrat / Centre-left"
        elif avg < 0.2:
            unified["overall_leaning"] = "Neutral / Centrist"
        elif avg < 0.6:
            unified["overall_leaning"] = "Mildly Republican / Centre-right"
        else:
            unified["overall_leaning"] = "Strongly Republican / Right-leaning"

    def analyse(self, comments: list) -> dict:
        chunks  = [comments[i:i + CHUNK_SIZE] for i in range(0, len(comments), CHUNK_SIZE)]
        unified = self._init_unified()

        print(f"\n🔍 Analysing {len(comments)} comments in {len(chunks)} chunks of {CHUNK_SIZE}...\n")

        for chunk in tqdm(chunks, desc="🧠 Analysing chunks", unit="chunk"):
            prompt = self._build_prompt(chunk)
            raw    = self._call_ollama(prompt)
            if raw:
                parsed = self._parse_response(raw)
                self._append_to_unified(parsed, unified)

        self._finalise(unified)
        return unified

    def print_summary(self, summary: dict):
        print("\n" + "="*60)
        print("📊  POLITICAL COMMENT ANALYSIS SUMMARY")
        print("="*60)
        print(f"  Total comments analysed : {summary['total_comments_analysed']}")
        print(f"  Overall political score : {summary['overall_political_score']:+.4f}  (-1=Democrat, +1=Republican)")
        print(f"  Overall leaning         : {summary['overall_leaning']}")
        print(f"  Sarcasm detected        : {summary['sarcasm_detected_count']} comments")
        print("\n  📂 Category Breakdown:")
        for cat, items in summary["category_breakdown"].items():
            print(f"     {cat:<25} {len(items)} comments")
        print("\n  😊 Sentiment Breakdown:")
        for sentiment, count in summary["sentiment_breakdown"].items():
            print(f"     {sentiment:<10} {count}")
        print("\n  🗳️  Score Distribution:")
        dist  = summary["score_distribution"]
        total = summary["total_comments_analysed"] or 1
        for label, count in dist.items():
            bar = "█" * int((count / total) * 30)
            print(f"     {label:<22} {bar} {count}")
        print("="*60 + "\n")


def main():
    # ── Step 1: Fetch Comments ────────────────────────────────────────────────
    fetcher   = YouTubeCommentsFetcher()
    video_url = input("🚀 Enter YouTube URL: ").strip()
    comments  = fetcher.fetch(video_url)

    if not comments:
        print("❌ No comments fetched. Please check your YouTube API key and URL.")
        return

    # ── Step 2: Analyse Comments ──────────────────────────────────────────────
    analyser = PoliticalCommentAnalyser()
    summary  = analyser.analyse(comments)

    # ── Step 3: Save Results ──────────────────────────────────────────────────
    with open("raw_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)
    logging.info("✅ Full results saved to raw_results.json")

    # ── Step 4: Print Summary ─────────────────────────────────────────────────
    analyser.print_summary(summary)


if __name__ == "__main__":
    main()