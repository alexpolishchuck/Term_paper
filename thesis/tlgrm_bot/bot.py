import os
import random
import numpy as np
from keras.saving.saving_api import load_model
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import ContextTypes, ConversationHandler
from models.models_consts import model_name, folder_name, X_user_likes_song, X_user_doesnt_like_song
from models.train_models import build_user_recommendation_model, train_user_recommendation_model, features_extractor, \
    predict_genre
from tlgrm_bot.conversation_states import Conversation_state


class bot_impl:

    @staticmethod
    async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
        print(f'Update {update} caused error {context.error}')

    @staticmethod
    async def recommend_song(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        try:
            recomm_model_folder = folder_name.MODELS_FOLDER.value + str(update.effective_chat.id) + "/"

            recommendation_model = load_model(recomm_model_folder
                                              + model_name.USER_RECOMMENDATION_MODEL.value)

            genre = predict_genre(recommendation_model, X_user_likes_song)

            context.user_data["genre"] = genre

            await update.message.reply_text(
                "Your favourite genre is " + genre.name
            )

            path = folder_name.TRAIN_DATA_FOLDER.value + "/" + genre.name

            for root, dirs, files in os.walk(path):
                size = len(files)

                pos = random.randint(0, size - 1)

                filename = files[pos]

                file_path = path + "/" + filename

                audio_file = open(file_path, 'rb')
                await context.bot.send_audio(
                    chat_id=update.effective_chat.id, audio=audio_file)
                break

            reply_keyboard = [["Yes", "No"]]

            await update.message.reply_text(
                "Did you like the audio we sent you?",
                reply_markup=ReplyKeyboardMarkup(
                    reply_keyboard, one_time_keyboard=True, input_field_placeholder="Tell us:"
                ),
            )

        except Exception as ex:
            print("recommend_song. " + str(ex))
            return ConversationHandler.END

        return Conversation_state.on_user_response.value

    @staticmethod
    async def on_user_response(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        await update.message.reply_text(
            "Thank you for the feedback!",
            reply_markup=ReplyKeyboardRemove(),
        )

        recommendation_model_folder = folder_name.MODELS_FOLDER.value + str(update.effective_chat.id) + "/"

        genre = context.user_data["genre"]

        if(update.message.text == "Yes"):
            train_user_recommendation_model(X_user_likes_song,
                                            [[genre.value]],
                                            bot_impl.load_recommendation_model(update.effective_chat.id),
                                            recommendation_model_folder)
        elif(update.message.text == "No"):
            train_user_recommendation_model(X_user_doesnt_like_song,
                                            [[genre.value]],
                                            bot_impl.load_recommendation_model(update.effective_chat.id),
                                            recommendation_model_folder)
        return ConversationHandler.END

    @staticmethod
    async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
        return ConversationHandler.END
    @staticmethod
    async def handle_audio_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "Parsing file."
        )

        try:

            if update.message.document is not None:

                print(update.message.document)
                file_info = await context.bot.get_file(update.message.document.file_id)
                filename = update.message.document.file_name

            elif update.message.audio is not None:

                print(update.message.audio)
                file_info = await context.bot.get_file(update.message.audio.file_id)
                filename = update.message.audio.file_name

            dir_path = bot_impl.downloaded_files_dir + str(update.effective_chat.id)

            try:
                os.mkdir(dir_path)
            except Exception as ex:
                print("Folder already exist or some other error occurred")

            file_path = dir_path + "/" + filename

            await file_info.download_to_drive(custom_path=file_path)

            recommendation_model_folder = folder_name.MODELS_FOLDER.value + str(update.effective_chat.id) + "/"
            recommendation_model = bot_impl.load_recommendation_model(update.effective_chat.id)

            genre_classifier_model = load_model(
                folder_name.MODELS_FOLDER.value + model_name.GENRE_CLASSIFIER_MODEL.value)

            X = features_extractor(file_path)
            X = X[np.newaxis, np.newaxis, ...]
            y = predict_genre(genre_classifier_model, X)

            train_user_recommendation_model(X_user_likes_song,
                                            [[y.value]],
                                            recommendation_model,
                                            recommendation_model_folder)

            await update.message.reply_text(
                "The filename:" + filename + "; Genre: " + y.name
            )

        except Exception as ex:
            print(str(ex))
            await update.message.reply_text(
                "Something went wrong."
            )

        try:
            os.remove(file_path)
        except Exception as ex:
            print("Couldn't remove file: " + filename)

    @staticmethod
    def load_recommendation_model(chat_id: int):
        recommendation_model_folder = folder_name.MODELS_FOLDER.value + str(chat_id) + "/"

        try:
            recommendation_model = load_model(recommendation_model_folder
                                              + model_name.USER_RECOMMENDATION_MODEL.value)
        except Exception as ex:
            print("There is no recommendation_model for the current chat, id = "
                  + str(chat_id))
            recommendation_model = build_user_recommendation_model()

        return recommendation_model


    API_KEY = '6130709421:AAHiuWp5Au4StNbhbnXl41jBg7ianqhd8iA'

    downloaded_files_dir = "downloaded_files/"
