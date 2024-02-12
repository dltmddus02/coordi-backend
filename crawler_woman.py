import urllib.request
import os
import csv, time, platform
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager

print("current directory :",os.getcwd())
# URL = 'https://laurant051.com/product/list.html?cate_no='
URL = 'https://meltingpixel.kr/product/list.html?cate_no='
# CATE_NO = map(str, range(80, 200))
CATE_NO = map(str, range(0, 200))
IMAGE_DIR = 'images/'
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

url_file = open('urls.csv', 'w', newline='')
url_csv = csv.writer(url_file)

if platform.system()=='Windows':
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--log-level=3')
    service = webdriver.ChromeService(executable_path=ChromeDriverManager().install())
    browser = webdriver.Chrome(options=options, service=service)
elif platform.system()=='Linux':
    options = webdriver.FirefoxOptions()
    options.add_argument('--headless')
    service = webdriver.FirefoxService(executable_path=GeckoDriverManager().install())
    browser = webdriver.Firefox(options=options, service=service)

cnt = 0
for cate_no in CATE_NO:
    browser.get(URL + cate_no)
    while True:
        img_elements = browser.find_elements(By.CSS_SELECTOR, 'img')
        # print(img_elements)
        cate_exists = False
        for img in img_elements:
            # print(img.get_property('class'))
            # if img.get_attribute('class') in ['thumber_1', 'thumber_2']:
            ancestor_elements = img.find_elements(By.XPATH, 'ancestor::*')
            has_thumbnail_ancestor = any('thumbnail'in ancestor.get_attribute('class') for ancestor in ancestor_elements if ancestor.get_attribute('class'))
            if not has_thumbnail_ancestor :
                continue
            cate_exists = True
            print("true")
            # print(img)
            img_url = img.get_attribute('src') # 이미지 url 가져오기
            print(img_url)
            urllib.request.urlretrieve(img_url, IMAGE_DIR + f'{cnt}.jpg')
            a = img.find_element(By.XPATH, '..')
            url_csv.writerow([cnt, a.get_attribute('href')])
            print(f'{cnt}.jpg')
            cnt = cnt + 1   

        if not cate_exists:
            print(f'cate_no {cate_no} does not exist')
            break

        next_btn = browser.find_element(By.XPATH, '//*[@id="container"]/div[2]/a[2]')
        next_btn.click()
        if browser.current_url.endswith('#none'):
            print('search finished.')
            break
        print('next page')
        break


browser.close()
url_file.close()