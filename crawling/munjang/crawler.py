import os
import time
import datetime
import pandas as pd

from tqdm.auto import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By

BASE_PATH = ''
MUNJANG_TEEN_URL = ''
MUNJANG_WEBZINE_URL = ''

DRIVER_PATH = os.path.join(BASE_PATH, '')
DRIVER_IS_VISIBLE = False


def crawl_webzine(category='novel'):
    # Parameter Setting
    save_dir = os.path.join(BASE_PATH, f'{category}')    
    target_url = f'{MUNJANG_WEBZINE_URL}/archives/category/{category}'

    # DIRECTORY
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # WEB DRIVER
    options = webdriver.ChromeOptions()
    options.add_argument('--log-level=3')
    
    if not DRIVER_IS_VISIBLE:
        options.add_argument('headless')

    driver = webdriver.Chrome(executable_path=DRIVER_PATH, options=options)
    driver.get(target_url)

    # CRAWLING
    page_begin, page_end = 1, 10000

    for i in tqdm(range(page_begin, page_end + 1), desc=f'MUNJANG webzine {category}'):
        driver.get(f'{target_url}/page/{i}')

        article_ids = driver.find_elements(by=By.CSS_SELECTOR, value='article')
        article_ids = [i.get_attribute('id') for i in article_ids]

        result = {'title': list(), 'content': list()}

        for id in article_ids:
            # title
            article = driver.find_elements(by=By.ID, value=id)[0]
            title = article.find_elements(by=By.CSS_SELECTOR, value='div > div.post_content > div.post_title > a')[0]
            title_text = title.text
            title.click(); time.sleep(0.5)

            # content
            content = driver.find_elements(by=By.CSS_SELECTOR, value='.entry-content')[0].text                
            
            if not title_text and not content:
                print(f'Last page is {i}. The loops will be ended.')
                print(f'datetime: {datetime.datetime.now()}')
                break

            else:
                result['title'].append(title_text)
                result['content'].append(content)

            driver.back(); time.sleep(0.5)

        result_df = pd.DataFrame(result)
        result_df.to_csv(os.path.join(save_dir, f'page_{i}.csv'), index=False, encoding='utf-8')

    print('All crawling is done. Driver will quit.')
    driver.quit()
    

def crawl_geultin(category, begin_at=1):
    # Parameter Setting
    save_dir = os.path.join(BASE_PATH, f'{category}')    
    target_url = f'{MUNJANG_TEEN_URL}/archives/category/write/{category}'

    # DIRECTORY
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # WEBDRIVER
    options = webdriver.ChromeOptions()
    options.add_argument('--log-level=3')
    if not DRIVER_IS_VISIBLE:
        options.add_argument('headless')

    driver = webdriver.Chrome(executable_path=DRIVER_PATH, options=options)
    driver.get(target_url)

    page_begin, page_end = 0 + begin_at, 10000

    for i in tqdm(range(page_begin, page_end + 1), desc=f'MUNJANG geultin {category}'):
        driver.get(f'{target_url}/page/{i}')

        article_ids = driver.find_elements(by=By.CSS_SELECTOR, value='article')
        article_ids = [i.get_attribute('id') for i in article_ids]

        result = {'title': list(), 'content': list()}

        for id in article_ids:
            # title
            article = driver.find_elements(by=By.ID, value=id)[0]
            title = article.find_elements(by=By.CSS_SELECTOR, value='div > div.post_content > div.post_title > a')[0]
            title_text = title.text
            title.click(); time.sleep(0.5)

            # content
            content = driver.find_elements(by=By.CSS_SELECTOR, value='.entry-content')[0].text

            if title_text == '':
                print(f'Last page is {i}. The loops will be ended.')
                print(f'datetime: {datetime.datetime.now()}')
                break

            else:
                result['title'].append(title_text)
                result['content'].append(content)

            driver.back(); time.sleep(0.5)

        result_df = pd.DataFrame(result)
        result_df.to_csv(os.path.join(save_dir, f'page_{i}.csv'), index=False, encoding='utf-8')

    print('All crawling is done. Driver will quit.')
    driver.quit()


if __name__ == '__main__':
    # crawl_webzine()
    # crawl_geultin(category='life')
    crawl_geultin(category='story')
