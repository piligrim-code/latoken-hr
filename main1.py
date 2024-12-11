import os
import asyncio
import logging
import json
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import BaseFilter, StateFilter
from aiogram.filters.command import Command
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import KeyboardButton,ContentType, ReplyKeyboardMarkup
from aiogram.fsm.context import FSMContext
from llm import query_chatgpt, search_and_respond
from langchain.prompts import ChatPromptTemplate
from openai import OpenAI
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
from vector import get_embedding_function

with open('prompts.json', 'r', encoding='utf-8') as file:
    prompts = json.load(file)

CHROMA_PATH = "chroma"

class TestStates(StatesGroup):
    waiting_for_answer_1 = State()
    waiting_for_answer_2 = State()
    waiting_for_answer_3 = State()
    waiting_for_answer_1_feedback = State()
    waiting_for_answer_2_feedback = State()
    waiting_for_answer_3_feedback = State()

load_dotenv()

TOKEN = os.getenv("BOT_TOKEN")

logging.basicConfig(level=logging.INFO)

# Объект бота
bot = Bot(TOKEN)
# Диспетчер
dp = Dispatcher()

def make_reply_keyboard(button_name_rows):
    button_rows = []
    for button_name_row in button_name_rows:
        button_rows.append([KeyboardButton(text=text) for text in button_name_row])
    keyboard = ReplyKeyboardMarkup(keyboard=button_rows, resize_keyboard=True)
    return keyboard

@dp.message(Command("start"))
async def start_func(message: types.Message, state: FSMContext):
    await state.clear()
    user_data = await state.get_data()
    messages = user_data.get("messages", [])
    messages.append({"role": "assistant", "content": "Добрый день! Что бы вы хотели??"})
    keyboard = make_reply_keyboard([["Тест на проверку знаний"]])
    await state.update_data(messages=messages)
    await message.answer("Добрый день! Я HR-бот компании LATOKEN. Задайте мне любой вопрос о компании и я обязательно помогу вам! Если вы хотите протестировать свои знания о компании, нажмите на кнопку Тест на проверку знаний", reply_markup=keyboard)

@dp.message(F.text == "Тест на проверку знаний")
async def test_knowledge(message: types.Message, state: FSMContext):
    # Starting the quiz
    await message.answer("Хорошо, я сейчас задам вам три вопроса, а вы попробуете на них ответить, вопросы будут задаваться подряд, после вашего ответа я расскажу правильный ли ваш ответ или нет, если не получится ответить с первого раза ничего страшного!")
    
    # Initialize a counter for questions asked
    await state.update_data(question_count=0)

    # Ask the first question
    question, correct_answer = await search_and_ask(message.text)
    
    # Save the answer in state
    await state.update_data(correct_answer_1=correct_answer)
    
    # Send the question to the user
    await message.answer(question)
    
    # Set state for waiting for answer 1
    await state.set_state(TestStates.waiting_for_answer_1)

# Handler for checking the answer to the first question
@dp.message(StateFilter(TestStates.waiting_for_answer_1))
async def check_answer_1(message: types.Message, state: FSMContext):
    user_data = await state.get_data()
    correct_answer = user_data.get("correct_answer_1")
    user_answer = message.text.strip()

    prompt2 = "\n".join(prompts['prompt2'])

    prompt_template = ChatPromptTemplate.from_template(prompt2)
    prompt = prompt_template.format(user_answer=user_answer, correct_answer=correct_answer)

    explanation = query_chatgpt(prompt)
    await message.answer(explanation)

    # Move to the next question immediately after feedback
    await ask_next_question(message, state)

# Function to handle asking the next question
async def ask_next_question(message, state):
    user_data = await state.get_data()
    question_count = user_data.get("question_count", 0) + 1
    
    # Update question count
    await state.update_data(question_count=question_count)
    
    if question_count == 1:
        # Ask the second question
        question, correct_answer = await search_and_ask(message.text)
        await state.update_data(correct_answer_2=correct_answer)
        await message.answer(question)
        await state.set_state(TestStates.waiting_for_answer_2)
    elif question_count == 2:
        # Ask the third question
        question, correct_answer = await search_and_ask(message.text)
        await state.update_data(correct_answer_3=correct_answer)
        await message.answer(question)
        await state.set_state(TestStates.waiting_for_answer_3)
    else:
        # End the quiz after three questions
        await message.answer("Тест завершен! Спасибо за участие!")
        await state.clear()

# Handler for checking the second question's answer
@dp.message(StateFilter(TestStates.waiting_for_answer_2))
async def check_answer_2(message: types.Message, state: FSMContext):
    user_data = await state.get_data()
    correct_answer = user_data.get("correct_answer_2")
    user_answer = message.text.strip()

    prompt2 = "\n".join(prompts['prompt2'])

    prompt_template = ChatPromptTemplate.from_template(prompt2)
    prompt = prompt_template.format(user_answer=user_answer, correct_answer=correct_answer)

    explanation = query_chatgpt(prompt)
    await message.answer(explanation)

    # Move to the next question immediately after feedback
    await ask_next_question(message, state)

# Handler for checking the third question's answer
@dp.message(StateFilter(TestStates.waiting_for_answer_3))
async def check_answer_3(message: types.Message, state: FSMContext):
    user_data = await state.get_data()
    correct_answer = user_data.get("correct_answer_3")
    user_answer = message.text.strip()

    prompt2 = "\n".join(prompts['prompt2'])

    prompt_template = ChatPromptTemplate.from_template(prompt2)
    prompt = prompt_template.format(user_answer=user_answer, correct_answer=correct_answer)

    explanation = query_chatgpt(prompt)
    await message.answer(explanation)

    # End the quiz after the third question
    await message.answer("Тест завершен! Спасибо за участие!")
    await state.clear()

# Function for searching and getting questions with answers (async version)
async def search_and_ask(query_text):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    question_prompt = f"На основе следующего контекста сформулируйте один актуальный вопрос исключительно о компании и её ценностях, задавай вопрос на языке который использует пользователь.\n{context_text}"
    questions_response = query_chatgpt(question_prompt)
    
    question = parse_question_from_response(questions_response)
    
    question_with_context_prompt = f"{context_text}\n\nQuestion: {question}\nAnswer:"
    answer_response = query_chatgpt(question_with_context_prompt)
    
    return question, answer_response

# Функция для обработки ответа и извлечения вопросов
def parse_question_from_response(response_text):
    lines = response_text.split('\n')
    for line in lines:
        if line.strip():  
            return line.strip()
    return "Sorry, no question generated." 


@dp.message(F.content_type == ContentType.TEXT)
async def message_reply(message: types.Message, state: FSMContext):
    user_data = await state.get_data()
    messages = user_data.get("messages", [])
    answer = search_and_respond(message.text)
    await message.answer(answer)

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())