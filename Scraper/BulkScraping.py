from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from pyquery import PyQuery as pq
import re
import os
import json

from .BaseScrapingModel import BaseScraper
from .PageScraping import PageScaraper

from models.analyzingANDtranslating.analyze_Trans_deps import A_TDeps


class BulkScraper(BaseScraper):
    def __init__(
        self,
        main_page_url: str = "https://www.ajnet.me",
        headless: bool = True,
        load_images: bool = True,
        user_agent: str = "",
        news_samples_path: str = "assets",
        a_tDeps: A_TDeps = None,
    ):
        super().__init__(headless, load_images, user_agent)
        self.a_tDeps      = a_tDeps
        self.page_scraper = PageScaraper(main_page_url, a_tDeps=a_tDeps)
        self.all_summaries: list[dict] = []   # accumulates per-article summaries for daily post

        os.makedirs(news_samples_path, exist_ok=True)
        self.file_path = os.path.join(news_samples_path, "news_sample.jsonl")

    # ── helpers ───────────────────────────────────────────────────────────────

    def extract_day(self, href: str):
        match = re.search(r'/news/\d+/\d+/(\d+)/', href)
        return int(match.group(1)) if match else None

    # ── core scrape ───────────────────────────────────────────────────────────

    def scrape_bulk(self, url: str, Day: bool = False, num_of_samples: int = 10):
        """
        Scrape articles.

        Day=True  → store to Mongo+Chroma, run analysis, accumulate summaries.
        Day=False → write to JSONL for fine-tuning, no analysis.
        """
        links = self.get_links(url, Day, num_of_samples)

        if Day:
            self.all_summaries = []   # reset before each daily run

        with open(self.file_path, "a", encoding="utf-8") as f:
            for i, (href, title, description, current_day) in enumerate(links):
                if Day:
                    summary = self.page_scraper.scrape(href, title, description, current_day, Day)
                    if summary is not None:
                        self.all_summaries.append(summary)
                else:
                    article  = self.page_scraper.scrape(href, title, description, current_day, Day)
                    article  = {"id": i + 1, **article}
                    f.write(json.dumps(article, ensure_ascii=False) + "\n")

    # ── daily post ────────────────────────────────────────────────────────────

    def publish_daily_post(self) -> str | None:
        """
        Build a daily-news post from self.all_summaries using the OpenAI LLM,
        send it to Telegram, and return the generated text.

        Call this explicitly from scraping.py after scrape_bulk() finishes.
        Returns None if there are no summaries or the LLM call fails.
        """
        if not self.all_summaries:
            print("[BulkScraper] No summaries to publish.")
            return None

        from helpers.Config import get_settings
        from stores.llm.LLM_Factory import LLMFactory
        import requests

        settings = get_settings()

        # ── build the user prompt ─────────────────────────────────────────────
        articles_block = ""
        for idx, s in enumerate(self.all_summaries, 1):
            bullet_points = "\n".join(f"  - {point}" for point in s["story_summary"])
            articles_block += (
                f"{idx}. {s['title']}\n"
                f"   Description: {s['description']}\n"
                f"   Key points:\n{bullet_points}\n"
                f"   Link: {s['url']}\n\n"
            )

        system_prompt = (
            "You are a professional news editor. "
            "Write a concise, engaging daily news digest in Arabic. "
            "Cover every story briefly, include each article's link as a clickable reference, "
            "and keep the total length under 3000 characters so it fits a Telegram message."
        )

        user_prompt = (
            f"Here are today's news stories:\n\n{articles_block}"
            "Write the daily digest post now."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

        # ── call OpenAI via LLMFactory ────────────────────────────────────────
        provider = LLMFactory.create(
            provider=settings.PROVIDERS,
            config={
                "api_key": settings.OPENAI_API_KEY,
                "api_url": settings.OPENAI_API_URL,
            },
        )
        provider.set_generation_model(model_id=settings.OPENAI_GENERATION_MODEL_ID)
        post_text = provider.generate_text(messages)

        if not post_text:
            print("[BulkScraper] LLM returned empty response for daily post.")
            return None

        # ── send to Telegram ──────────────────────────────────────────────────
        self._send_to_telegram(post_text, settings)

        return post_text

    def _send_to_telegram(self, text: str, settings) -> None:
        """
        Send text to the configured Telegram chat.
        Splits automatically if the message exceeds Telegram's 4096-char limit.
        """
        import requests

        bot_token = settings.BotToken
        chat_id   = settings.chatID
        api_url   = f"https://api.telegram.org/bot{bot_token}/sendMessage"

        # Split into ≤4096-char chunks on newlines where possible
        max_len = 4096
        chunks  = []
        while len(text) > max_len:
            split_at = text.rfind("\n", 0, max_len)
            if split_at == -1:
                split_at = max_len
            chunks.append(text[:split_at])
            text = text[split_at:].lstrip("\n")
        chunks.append(text)

        for chunk in chunks:
            try:
                resp = requests.post(
                    api_url,
                    json={
                        "chat_id":    chat_id,
                        "text":       chunk,
                        "parse_mode": "HTML",
                    },
                    timeout=15,
                )
                resp.raise_for_status()
            except Exception as e:
                print(f"[BulkScraper] Telegram send error: {e}")

    # ── link collector ────────────────────────────────────────────────────────

    def get_links(self, url: str, Day: bool = False, num_of_samples: int = 10):
        self.driver.get(url)

        links      = []
        seen       = set()
        target_day = None

        while True:
            doc      = pq(self.driver.page_source)
            articles = doc('section#news-feed-container article')

            for article in articles.items():
                href        = article('a.article-card__link').attr('href')
                title       = article('a.article-card__link span').text()
                description = article('p.article-card__excerpt span').text()

                if not href or (href, title, description) in seen:
                    continue

                seen.add((href, title, description))

                current_day = self.extract_day(href)

                if Day and target_day is None:
                    target_day = current_day

                if Day and current_day != target_day:
                    return links

                links.append((href, title, description, current_day))

                if len(links) >= num_of_samples:
                    return links

            # Try clicking "Show more"
            try:
                show_more_btn = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable(
                        (By.CSS_SELECTOR, 'button.show-more-button')
                    )
                )
                self.driver.execute_script("arguments[0].scrollIntoView();", show_more_btn)

                old_count = len(self.driver.find_elements(By.CSS_SELECTOR, "article"))
                show_more_btn.click()

                WebDriverWait(self.driver, 10).until(
                    lambda d: len(d.find_elements(By.CSS_SELECTOR, "article")) > old_count
                )
            except Exception:
                break

        return links