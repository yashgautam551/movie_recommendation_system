import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self._prepare_data()
        self._build_model()

    def _prepare_data(self):
        self.df["content"] = self.df["overview"] + " " + self.df["genres"]
        self.df["content"] = self.df["content"].fillna("")

    def _build_model(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["content"])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

    def recommend(self, movie_title, top_n=5):
        if movie_title not in self.df["title"].values:
            return ["Movie not found in database"]

        index = self.df[self.df["title"] == movie_title].index[0]
        similarity_scores = list(enumerate(self.similarity_matrix[index]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        recommendations = []
        for i in similarity_scores[1:top_n + 1]:
            recommendations.append(self.df.iloc[i[0]]["title"])

        return recommendations


if __name__ == "__main__":
    recommender = MovieRecommender("data/movies.csv")

    movie_name = "Inception"
    print(f"\n🎬 Recommendations for '{movie_name}':\n")

    results = recommender.recommend(movie_name)
    for movie in results:
        print("➡️", movie)
