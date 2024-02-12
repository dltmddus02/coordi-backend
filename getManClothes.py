import requests
from bs4 import BeautifulSoup
import os

def is_valid_filename(filename):
    invalid_chars = '<>:"/\\|?*'
    return not any(char in filename for char in invalid_chars)

url = "https://laurant051.com/product/list.html?cate_no=29/"
# 사용자 에이전트 설정. (로랑이가 크롤링 막아놔서.)
# headers 매개변수 추가해서 일반적인 웹 브라우저에 보내는 것과 유사한 사용자 에이전트 설정.
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

folder_path = 'downloaded_man_clothes_images'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# print(response)
for img in soup.find_all('img'):
    img_url = img.get('src')
    # print(img_url)
    if img_url.startswith('/'):
        # 상대경로인 경우 절대경로로 변환
        img_url = url + img_url
    
    img_name = img_url.split('/')[-1]
    if is_valid_filename(img_name):
        full_path = os.path.join(folder_path, img_name)

        img_data = requests.get(img_url).content
        with open(full_path, 'wb') as file:
            file.write(img_data)