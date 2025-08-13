# Eora LLM Bot

Простой сервис на FastAPI + HTML/CSS/JS, который отвечает на вопросы клиентов, используя материалы с сайта `eora.ru`. В ответ вставляются ссылки на использованные источники в виде меток `[N]`.


## Требования
- Python 3.10+
- Доступ в интернет для загрузки кейсов и обращения к API

## Настройка API-ключа
Ключ можно получить здесь:
- https://openrouter.ai/openai/gpt-4o-mini/api

Для локального запуска нужно создать `.env` рядом с `main.py`:
```
OPENROUTER_API_KEY=[ваш_токен]
```

## Установка и запуск
Из корня проекта:
```
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
Откройте: http://127.0.0.1:8000/

## Пример вопроса
```
Что вы можете сделать для ритейлеров?
```
Бот сформирует краткий ответ и добавит метки источников `[1]`, `[2]`, ... со ссылками на соответствующие кейсы.
