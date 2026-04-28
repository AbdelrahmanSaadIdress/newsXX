from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from pyquery import PyQuery as pq
import re
import os
import json

from .BaseScrapingModel import BaseScraper
from .PageScraping import PageScaraper

class BulkScraper(BaseScraper):
    def __init__(self,main_page_url:str="https://www.ajnet.me", headless=True, load_images=True, user_agent='', news_samples_path="assets"):
        super().__init__(headless, load_images, user_agent)
        self.page_scraper = PageScaraper(main_page_url)

        os.makedirs(news_samples_path, exist_ok=True)
        self.file_path = os.path.join(news_samples_path, "news_sample.jsonl")

    def extract_day(self, href):
        match = re.search(r'/news/\d+/\d+/(\d+)/', href)
        return int(match.group(1)) if match else None

    def scrape_bulk(self, url, Day: bool = False, num_of_samples=10):
        links = self.get_links(url, Day, num_of_samples)
                
        with open(self.file_path, "a", encoding="utf-8") as f:
            articles = [] if Day else None
            for i in range(len(links)):
                href, title, description, current_day = links[i]
                if Day:
                    self.page_scraper.scrape(href, title, description, current_day, Day)
                else:
                    article = self.page_scraper.scrape(href, title, description, current_day, Day)
                    article = {"id": i+1, **article}
                    json_line = json.dumps(article, ensure_ascii=False)
                    f.write(json_line + "\n")   

    def get_links(self, url, Day: bool = False, num_of_samples=10):
        self.driver.get(url)

        links = []
        seen = set()
        target_day = None

        while True:
            doc = pq(self.driver.page_source)
            articles = doc('section#news-feed-container article')

            for article in articles.items():
                href = article('a.article-card__link').attr('href')
                
                title = article('a.article-card__link span').text()
                description = article('p.article-card__excerpt span').text()

                if not href or (href, title, description) in seen:
                    continue

                seen.add((href, title, description))

                # Extract day
                current_day = self.extract_day(href)

                # Set target day (first article only)
                if Day and target_day is None:
                    target_day = current_day

                # 🔴 STOP CONDITION 1: day changed
                if Day and current_day != target_day:
                    return links

                links.append((href, title, description, current_day))

                # 🔴 STOP CONDITION 2: reached limit
                if len(links) >= num_of_samples:
                    return links

            # Try clicking "Show more"
            try:
                show_more_btn = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable(
                        (By.CSS_SELECTOR, 'button.show-more-button')
                    )
                )

                self.driver.execute_script(
                    "arguments[0].scrollIntoView();", show_more_btn
                )

                old_count = len(self.driver.find_elements(By.CSS_SELECTOR, "article"))

                show_more_btn.click()

                WebDriverWait(self.driver, 10).until(
                    lambda d: len(d.find_elements(By.CSS_SELECTOR, "article")) > old_count
                )

            except Exception:
                break  # no more articles

        return links