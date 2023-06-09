from enum import Enum

X_user_likes_song = [[1, 0]]
X_user_doesnt_like_song = [[0, 1]]
class model_name(Enum):
    GENRE_CLASSIFIER_MODEL = "genre_classifier_model.h5"
    USER_RECOMMENDATION_MODEL = "user_recommendation_model.h5"


class folder_name(Enum):
    MODELS_FOLDER = "../trained_models/"
    TRAIN_DATA_FOLDER = "../Data/genres_original"
