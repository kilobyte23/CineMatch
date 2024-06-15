class MovieRecommendationSystem:
    def __init__(self):
        self.movies = []

    def add_movie(self, title, genre, rating):
        movie = {'title': title, 'genre': genre, 'rating': rating}
        self.movies.append(movie)
        print(f"Movie '{title}' added successfully.")

    def search_movie(self, keyword):
        results = [movie for movie in self.movies if keyword.lower() in movie['title'].lower() or keyword.lower() in movie['genre'].lower()]
        if results:
            print(f"Search results for '{keyword}':")
            for movie in results:
                print(f"Title: {movie['title']}, Genre: {movie['genre']}, Rating: {movie['rating']}")
        else:
            print(f"No movies found for '{keyword}'.")

    def recommend_top_movies(self, N):
        sorted_movies = sorted(self.movies, key=lambda x: x['rating'], reverse=True)
        top_movies = sorted_movies[:N]
        print(f"Top {N} movies based on rating:")
        for movie in top_movies:
            print(f"Title: {movie['title']}, Genre: {movie['genre']}, Rating: {movie['rating']}")

    def delete_movie(self, title):
        for movie in self.movies:
            if movie['title'].lower() == title.lower():
                self.movies.remove(movie)
                print(f"Movie '{title}' deleted successfully.")
                return
        print(f"No movie found with title '{title}'.")

# Example usage:
if __name__ == "__main__":
    system = MovieRecommendationSystem()
    system.add_movie("The Shawshank Redemption", "Drama", 9.3)
    system.add_movie("The Godfather", "Crime", 9.2)
    system.add_movie("The Dark Knight", "Action", 9.0)
    system.add_movie("Pulp Fiction", "Crime", 8.9)
    system.add_movie("The Lord of the Rings: The Return of the King", "Fantasy", 8.9)
    system.add_movie("Forrest Gump", "Drama", 8.8)
    system.add_movie("Inception", "Sci-Fi", 8.8)
    system.add_movie("Fight Club", "Drama", 8.8)
    system.add_movie("The Matrix", "Sci-Fi", 8.7)
    system.add_movie("Goodfellas", "Crime", 8.7)
    system.add_movie("The Lord of the Rings: The Fellowship of the Ring", "Fantasy", 8.8)
    system.add_movie("The Lord of the Rings: The Two Towers", "Fantasy", 8.7)
    system.add_movie("Star Wars: Episode V - The Empire Strikes Back", "Sci-Fi", 8.7)
    system.add_movie("Interstellar", "Sci-Fi", 8.6)
    system.add_movie("City of God", "Crime", 8.6)
    system.add_movie("Spirited Away", "Animation", 8.6)
    system.add_movie("Saving Private Ryan", "War", 8.6)
    system.add_movie("The Green Mile", "Drama", 8.6)
    system.add_movie("Gladiator", "Action", 8.5)
    system.add_movie("The Lion King", "Animation", 8.5)

    while True:
        print("\nMovie Recommendation System")
        print("1. Add a new movie")
        print("2. Search for movies by title or genre")
        print("3. Recommend top N movies based on rating")
        print("4. Delete a movie")
        print("5. Exit")
        
        choice = input("Enter your choice: ")

        if choice == "1":
            title = input("Enter movie title: ")
            genre = input("Enter movie genre: ")
            rating = float(input("Enter movie rating (0-10): "))
            system.add_movie(title, genre, rating)
        elif choice == "2":
            keyword = input("Enter keyword to search (title or genre): ")
            system.search_movie(keyword)
        elif choice == "3":
            N = int(input("Enter number of top movies to recommend: "))
            system.recommend_top_movies(N)
        elif choice == "4":
            title = input("Enter the title of the movie to delete: ")
            system.delete_movie(title)
        elif choice == "5":
            print("Exiting the system.")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")
