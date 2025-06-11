from enum import Enum

from tempfile import TemporaryFile
from matplotlib import pyplot as plt
import pandas as pd
import requests
import os
from telegram.ext import ApplicationBuilder

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardRemove
)

from telegram.ext import (
    CommandHandler,
    MessageHandler,
    filters,
    CallbackQueryHandler,
    ContextTypes,
    ConversationHandler,
)
from telegram.constants import ParseMode

class ForecastState(str, Enum):
    CHOOSE_FREQ = "freq"
    CHOOSE_TYPE = "type"
    CHOOSE_MODEL = "model"
    GET_HORIZON = "horizon"
    GET_RESULT = "result"


# Цепочка выбора нужного прогноза

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Starts the conversation and asks the user about their gender."""
    keyboard = [
        [
            InlineKeyboardButton("Приемка", callback_data="incoming"),
            InlineKeyboardButton("Отгрузка", callback_data="outgoing"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "Выберите целевую переменную",
        reply_markup=reply_markup,
    )
    return ForecastState.CHOOSE_FREQ

async def choose_freq(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Stores the selected gender and asks for a picture."""
    query = update.callback_query
    await query.answer()
    keyboard = [
        [InlineKeyboardButton("По дням", callback_data="day")],
        [InlineKeyboardButton("По часам", callback_data="hour")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    context.user_data["target"] = query.data
    await query.edit_message_text(
        "Выберите частоту прогноза",
        reply_markup=reply_markup,
    )
    return ForecastState.CHOOSE_TYPE

async def choose_type(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Stores the selected gender and asks for a picture."""
    query = update.callback_query
    await query.answer()
    keyboard = [
        [InlineKeyboardButton("ARMA", callback_data="arima")],
        [InlineKeyboardButton("ML", callback_data="sklearn")],
        [InlineKeyboardButton("DL", callback_data="torch")],
        [InlineKeyboardButton("Hybrid", callback_data="hybrid")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    context.user_data["frequency"] = query.data
    await query.edit_message_text(
        "Выберите тип модели",
        reply_markup=reply_markup,
    )
    return ForecastState.CHOOSE_MODEL

def model_keybord(model_type):
    model_dict = {
        "arima" : ["ARIMA", "SARIMA",],
        "sklearn" : ["Lasso", "RandomForest", "GradientBoosting", "CatBoost"],
        "torch" : ["GRU", "LSTM", "SEQ2SEQ", "SEQ2SEQ_full", "SEQ2SEQ_attention", "SEQ2SEQ_attention_full"],
        "hybrid" : ["GRU", "LSTM", "SEQ2SEQ", "SEQ2SEQ_full", "SEQ2SEQ_attention", "SEQ2SEQ_attention_full"],    
    }
    keyboard = [
            [InlineKeyboardButton(model, callback_data=model)] for model in model_dict[model_type]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    return reply_markup

async def choose_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Stores the selected gender and asks for a picture."""
    query = update.callback_query
    await query.answer()
    model_type = query.data
    context.user_data["model_type"] = model_type
    reply_markup = model_keybord(model_type)
    await query.edit_message_text(
        "Выберите модель",
        reply_markup=reply_markup,
    )
    return ForecastState.GET_HORIZON

async def get_horizon(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Stores the selected gender and asks for a picture."""
    query = update.callback_query
    await query.answer()
    context.user_data["model_name"] = query.data
    await query.edit_message_text(
        "Введите горизонт прогноза (количество шагов вперед)",
    )
    return ForecastState.GET_RESULT

def plot_result(responce, file, width, model):
    forecast = responce["forecast"]
    targets = responce["targets"]
    dates = responce["dates"]

    plt.figure(figsize=(10, 5))
    # ширина баров в зависимости от количества данных targets
    plt.bar(pd.to_datetime(dates), targets, label="Target", alpha=0.6, width=width)
    plt.bar(pd.to_datetime(dates)[-len(forecast):], forecast, label=model, alpha=0.6, width=width)
    plt.xticks(rotation=45)

    plt.ylabel("Volume, m$^3$")
    plt.xlabel("date")
    plt.legend()
    plt.tight_layout() 
    plt.savefig(file)
    plt.close()
    

async def get_result(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Stores the selected gender and asks for a picture."""
    horizon = update.message.text
    if not horizon.isdigit():
        await update.message.reply_text("Пожалуйста, введите число.")
        return ForecastState.GET_RESULT
    
    context.user_data["horizon"] = int(horizon)

    # Отправляем запрос к API
    try:
        response = requests.post(
            f"{os.getenv('BACKEND_URL')}/forecast",
            json= context.user_data
        )
        response.raise_for_status()  # Проверка на ошибки HTTP
        forecast_data = response.json()

        # Форматируем результат для отправки пользователю
        forecast_text = "Ваш прогноз готов"
        await update.message.reply_text(forecast_text)
        with TemporaryFile() as temp_file:
            width = 0.5 if context.user_data["frequency"] == "day"  else 0.05

            plot_result(forecast_data, temp_file, width,  context.user_data["model_name"])
            temp_file.seek(0)
            await update.message.reply_photo(photo=temp_file)

    except Exception as e:
        await update.message.reply_text(f"Произошла непредвиденная ошибка: {e}")

    # Завершаем диалог
    return ConversationHandler.END
    
    

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels and ends the conversation."""
    user = update.message.from_user
    await update.message.reply_text(
        "Bye! I hope we can talk again some day.", reply_markup=ReplyKeyboardRemove()
    )
    return ConversationHandler.END

forecast_handler = ConversationHandler(
    entry_points=[CommandHandler("forecast", start)],
    states={
        ForecastState.CHOOSE_FREQ: [CallbackQueryHandler(choose_freq)],
        ForecastState.CHOOSE_TYPE: [CallbackQueryHandler(choose_type)],
        ForecastState.CHOOSE_MODEL: [CallbackQueryHandler(choose_model)],
        ForecastState.GET_HORIZON: [CallbackQueryHandler(get_horizon)],
        ForecastState.GET_RESULT: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_result)],
    },
    fallbacks=[CommandHandler("cancel", cancel)],
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
  
    await update.message.reply_text(
        "Привет, чтобы получить прогноз, используйте команду /forecast"
    )
    
if __name__ == "__main__":
    application = ApplicationBuilder().token(os.getenv("TOKEN")).read_timeout(100).write_timeout(100).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(forecast_handler)

    application.run_polling(allowed_updates=Update.ALL_TYPES)
