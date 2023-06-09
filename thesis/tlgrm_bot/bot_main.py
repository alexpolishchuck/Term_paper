from telegram.ext import Application, CommandHandler, MessageHandler, filters, ConversationHandler

from bot import bot_impl
from tlgrm_bot.conversation_states import Conversation_state

if __name__ == '__main__':
    app = Application.builder().token(bot_impl.API_KEY).build()

    app.add_error_handler(bot_impl.error)

    app.add_handler(MessageHandler(filters.AUDIO | filters.Document.WAV | filters.Document.MP3,
                                   bot_impl.handle_audio_upload))

   # app.add_handler(CommandHandler('recommend', bot_impl.recommend_song))

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("recommend", bot_impl.recommend_song)],
        states={
            Conversation_state.on_user_response.value: [MessageHandler(filters.Regex("^(Yes|No)$"),
                                                                       bot_impl.on_user_response)],
        },
        fallbacks=[CommandHandler("cancel", bot_impl.cancel)],
    )

    app.add_handler(conv_handler)

    app.run_polling(poll_interval=5)
