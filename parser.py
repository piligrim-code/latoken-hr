import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime

def parse_url(url, file_name):
    try:
        # Отправляем GET-запрос на указанный URL
        response = requests.get(url)

        # Проверяем, успешно ли выполнен запрос
        if response.status_code == 200:
            # Создаем объект BeautifulSoup для парсинга HTML
            soup = BeautifulSoup(response.text, 'html.parser')

            # Извлекаем весь текст со страницы
            text = soup.get_text()

            # Сохраняем результат в текстовый файл
            save_text_to_file(text, file_name)
        else:
            print(f"Ошибка: не удалось получить страницу {url}. Код ответа: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Ошибка запроса для {url}: {e}")

def save_text_to_file(text, file_name):
    # Убедимся, что папка data существует
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Создаем путь для файла
    file_path = os.path.join('data', file_name)

    # Записываем текст в файл
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Текст сохранен в {file_path}")

for i, url in enumerate(urls):
    # Генерация уникального имени файла с номером и текущей временной меткой
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')  # Используем временную метку для уникальности
    file_name = f'parsed_page_{timestamp}_{i+1}.txt'  # Уникальное имя для каждого файла

    parse_url(url, file_name)