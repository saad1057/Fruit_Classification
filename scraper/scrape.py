import os
import time
from icrawler.builtin import GoogleImageCrawler

def scrape_images(fruit_name, max_num=300):
    # Set the directory to save images
    save_dir = os.path.join("../dataset", fruit_name.replace(" ", "_"))
    
    # Skip scraping if already enough images exist
    if os.path.exists(save_dir) and len(os.listdir(save_dir)) >= 50:
        print(f"✅ Already scraped: {fruit_name} ({len(os.listdir(save_dir))} images)")
        return

    os.makedirs(save_dir, exist_ok=True)

    # Set up the Google image crawler
    google_crawler = GoogleImageCrawler(
        storage={"root_dir": save_dir},
        log_level="INFO"
    )

    # Start crawling
    google_crawler.crawl(
        keyword=fruit_name,
        max_num=max_num,
        filters={
            "type": "photo",     # Only photos (not drawings or clipart)
            "size": "medium"     # Medium size for balance between quality and speed
        }
    )

# ✅ List of common fruits
fruits = [
    "apple fruit images",
    "banana fruit images",
    "orange fruit images",
    "strawberry fruit images",
    "mango fruit images",
    "peach fruit images",
    "grapes fruit images",
    "pineapple fruit images",
    "watermelon fruit images",
    "kiwi fruit images"
]

# ✅ Scrape images for each fruit
for fruit in fruits:
    print(f"\n🍎 Scraping: {fruit}")
    scrape_images(fruit, max_num=300)
    time.sleep(3)  # Be polite — delay between each fruit
