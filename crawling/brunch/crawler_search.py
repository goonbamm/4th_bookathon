import os
import time
import numpy as np
import pandas as pd

from urllib import parse
from tqdm.auto import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

SEARCH_TEXT = ''

BASE_PATH = ''
BRUNCH_URL = ''

DRIVER_PATH = os.path.join(BASE_PATH, '')
DRIVER_IS_VISIBLE = False

SCROLL_PAUSE_TIME = 1
NUM_PER_FILE = 10

LIMIT = 100000


def crawl_by_search_text(search_text):
    save_dir = os.path.join(BASE_PATH, f'brunch/{search_text}')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir) 

    # WEB DRIVER
    options = webdriver.ChromeOptions()
    if not DRIVER_IS_VISIBLE:
        options.add_argument('headless')

    driver = webdriver.Chrome(executable_path=DRIVER_PATH, options=options)
    
    driver.maximize_window()
    driver.implicitly_wait(10)
    
    driver.get(BRUNCH_URL)
    
    a = driver.find_elements(
        by=By.CSS_SELECTOR,
        value=f'#btnServiceMenuSearch'
        )[0]
    a.click()
    driver.find_element("id", "txt_search").send_keys(SEARCH_TEXT)
    driver.find_element("id", "txt_search").send_keys(Keys.ENTER)
    driver.implicitly_wait(10)
    driver.switch_to.window(driver.window_handles[-1])
    driver.get_window_position(driver.window_handles[-1])

    start_page = 1
    last_height = driver.execute_script('return document.body.scrollHeight')

    while True:
        result = {
            'title': list(), 'subtitle': list(), 'content': list(), 'like': list(), 
            'keywords': list(), 'comment': list(), 'author': list(), 'subscribe': list()
            }
        
        print('brunch: {} ~ {}'.format(start_page, start_page + NUM_PER_FILE - 1))
        for page in range(start_page, start_page + NUM_PER_FILE):

            while len(driver.window_handles) > 1:
                driver.switch_to.window(driver.window_handles[-1])
                driver.close()
                driver.implicitly_wait(10)
                driver.switch_to.window(driver.window_handles[-1])                
                
            acting_point = driver.find_elements(
                by=By.CSS_SELECTOR,
                value=f'div.wrap_article_list > ul > li:nth-child({page}) > a'
                )[0]
                   
            driver.execute_script("arguments[0].click();", acting_point)
            driver.implicitly_wait(10)
            driver.switch_to.window(driver.window_handles[-1])
            driver.get_window_position(driver.window_handles[-1])
                                
            # title
            title = driver.find_elements(
                by=By.CSS_SELECTOR,
                value='h1.cover_title'
                )[0]
            result['title'].append(title.text)
            
            # subtitle
            subtitle = driver.find_elements(
                by=By.CSS_SELECTOR,
                value='p.cover_sub_title'
                )[0]
        
            result['subtitle'].append(subtitle.text)            

            # content
            content = driver.find_elements(
                by=By.CSS_SELECTOR,
                value='div.wrap_body_frame> div:first-child'
                )[0]
            result['content'].append(content.text)
            
            # like
            like = driver.find_elements(
                by=By.CSS_SELECTOR,
                value='span.f_l.text_like_count.text_default.text_with_img_ico.ico_likeit_like.\#like'
                )[0].text
            result['like'].append(like if like != '' else '0')

            # keywords
            try:
                keywords = driver.find_elements(
                by=By.CSS_SELECTOR,
                value='ul.list_keyword'
                )[0]
                result['keywords'].append(keywords.text)
                
            except:
                result['keywords'].append('')

            # comment
            try:
                comment = driver.find_elements(
                    by=By.CSS_SELECTOR,
                    value='span.f_l.text_comment_count.text_default.text_with_img_ico'
                    )[0].text
                result['comment'].append(comment if comment != '' else '0')

            except: # comment permission denied
                result['comment'].append('0')

            # author
            author = driver.find_elements(
                by=By.CSS_SELECTOR,
                value='strong.author_name'
                )[0]
            result['author'].append(author.text)

            # subscribe
            subscribe = driver.find_elements(
                by=By.CSS_SELECTOR,
                value='span.num_subscription'
                )[0].text
            result['subscribe'].append(subscribe if subscribe != '' else '0')

            driver.close()
            driver.switch_to.window(driver.window_handles[-1])
            driver.get_window_position(driver.window_handles[-1])
        
        result_df = pd.DataFrame(result)
        result_df.to_csv(os.path.join(save_dir, '{}_{}.csv'.format(start_page, start_page + NUM_PER_FILE - 1)),
                        index=False, encoding='utf-8')

        driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
        time.sleep(SCROLL_PAUSE_TIME)
        
        new_height = driver.execute_script('return document.body.scrollHeight')
        if new_height == last_height:
            print('Since new height == last height, we will stop it.')
            break
        
        last_height = new_height
        time.sleep(SCROLL_PAUSE_TIME)

        start_page += NUM_PER_FILE
        if start_page > LIMIT:
            print(f'We can get {start_page - 1} data! We will stop it :)')
            break

    print('All crawling is done. Driver will quit.')
    driver.quit()


if __name__ == '__main__':    
    print(f'Search Text: {SEARCH_TEXT}')
    crawl_by_search_text(SEARCH_TEXT)
    