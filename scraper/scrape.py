import os
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

# Set up Selenium WebDriver (example uses Chrome)
driver = webdriver.Chrome()

# Example: URL to scrape images from
target_url = 'https://example.com/fruits'
driver.get(target_url)

time.sleep(2)  # Wait for page to load

# Example: Find image elements (update selector as needed)
images = driver.find_elements(By.TAG_NAME, 'img')

# Create dataset directory if it doesn't exist
os.makedirs('../dataset', exist_ok=True)

for idx, img in enumerate(images):
    src = img.get_attribute('src')
    if src:
        # Download image (add your own download logic here)
        print(f'Download image {idx}: {src}')

# Clean up
driver.quit() 