def read_articles(filename):
    """
    Read articles from a file and return two arrays:
    - articles: array with one article in each location
    - headlines: array with the headlines of the articles (indexes match)
    
    Args:
        filename: Path to the file containing articles
        
    Returns:
        tuple: (articles, headlines) - two arrays with matching indexes
    """
    articles = []
    headlines = []
    
    with open(filename, 'r') as file:
        content = file.read()
        
    # Split content by double newlines to separate articles
    article_blocks = content.strip().split('\n\n')
    
    for block in article_blocks:
        if block.strip():
            lines = block.strip().split('\n')
            # First line should contain the headline
            if lines and lines[0].startswith('Headline:'):
                headline = lines[0].replace('Headline:', '').strip()
                # The rest is the article content
                article = '\n'.join(lines[1:]).strip()
                
                headlines.append(headline)
                articles.append(article)
    
    return articles, headlines


def main():
    """
    Main function that reads articles and displays them.
    """
    filename = 'develop.txt'
    
    # Call the function to read articles and get both arrays
    articles, headlines = read_articles(filename)
    
    # Display the results
    print(f"Total articles read: {len(articles)}")
    print("\n" + "="*70 + "\n")
    
    for i, (headline, article) in enumerate(zip(headlines, articles)):
        print(f"Article {i + 1}:")
        print(f"Headline: {headline}")
        print(f"Content: {article}")
        print("\n" + "-"*70 + "\n")


if __name__ == "__main__":
    main()
