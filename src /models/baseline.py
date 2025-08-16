"""
baseline.py

Модуль с простыми базовыми рекомендателями для построения бейзлайнов.
Содержит класс PopularRecommender, который формирует рекомендации на основе
глобального топа популярных товаров за последние N дней.
"""

import pandas as pd
from itertools import cycle, islice

class PopularRecommender:
    """
    Рекомендатель на основе популярности.

    Строит список топ-N популярных товаров за последние `days` дней
    и выдаёт их в качестве рекомендаций для пользователей.

    Параметры
    ---------
    top_k : int, default=50
        Сколько самых популярных товаров хранить во внутреннем списке.
    days : int, default=30
        За какой временной период считать популярность (в днях).
    item_column : str, default='item_id'
        Название столбца в DataFrame с идентификаторами товаров.
    dt_column : str, default='date'
        Название столбца в DataFrame с датами событий.

    Атрибуты
    --------
    recommendations : list
        Список популярных товаров, рассчитанный при вызове `fit()`.
    """

    def __init__(self, top_k: int = 50, days: int = 30,
                 item_column: str = 'item_id', dt_column: str = 'date'):
        self.top_k = top_k
        self.days = days
        self.item_column = item_column
        self.dt_column = dt_column
        self.recommendations = []

    def fit(self, df: pd.DataFrame) -> None:
        """
        Рассчитать топ популярных товаров на основе переданного датасета.

        Параметры
        ---------
        df : pandas.DataFrame
            Датафрейм с данными о взаимодействиях пользователей.
            Должен содержать столбцы `item_column` и `dt_column`.

        Возвращает
        ----------
        None
            Заполняет атрибут self.recommendations списком популярных товаров.
        """
        min_date = df[self.dt_column].max().normalize() - pd.Timedelta(days=self.days)
        self.recommendations = (
            df.loc[df[self.dt_column] > min_date, self.item_column]
            .value_counts()
            .head(self.top_k)
            .index
            .to_numpy()
        )

    def recommend(self, users=None, N: int = 10):
        """
        Вернуть рекомендации для заданных пользователей.

        Параметры
        ---------
        users : list или None, default=None
            Список идентификаторов пользователей. Если None, возвращается топ-N
            без учёта пользователей.
        N : int, default=10
            Сколько рекомендаций отдать на пользователя.

        Возвращает
        ----------
        list или numpy.ndarray
            - Если users=None → numpy.ndarray из топ-N товаров.
            - Если задан список users → list, где каждому пользователю
              сопоставлен один и тот же список топ-N товаров.
        """
        recs = self.recommendations[:N]
        if users is None:
            return recs
        return list(islice(cycle([recs]), len(users)))


#Пример использования класса
if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path

    # путь к датасету
    data_path = Path(__file__).resolve().parents[2] / "dataset" / "preprocessed_datasets" / "preprocessed_interactions.pkl"

    # загружаем данные
    df = pd.read_pickle(data_path)
    print("Датасет загружен:", df.shape)
    print(df.head())

    # инициализация рекомендателя
    model = PopularRecommender(
        top_k=50,
        days=30,
        item_column="item_id",
        dt_column="start_date"
    )

    # обучение
    model.fit(df)

    # рекомендации (просто топ-10)
    print("Global top-10:", model.recommend(N=10))

    # рекомендации для нескольких пользователей
    sample_users = df["user_id"].drop_duplicates().sample(3, random_state=42).tolist()
    print("Пример пользователей:", sample_users)
    print("Recs for sample users:", model.recommend(users=sample_users, N=5))