import argparse


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Namespace: Parsed arguments containing the filename
    """
    parser = argparse.ArgumentParser(
        description='Read articles from a text file and display them.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'filename',
        type=str,
        help='Path to the file containing articles'
    )
    return parser.parse_args()


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
    
    try:
        with open(filename, 'r') as file:
            content = file.read()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return articles, headlines
        
    # Split content by empty lines (one or more newlines)
    lines = content.split('\n')
    current_block = []
    
    for line in lines:
        # If line is empty (after stripping), it's a separator
        if not line.strip():
            # Process the current block if it has content
            if current_block:
                # First line should contain the headline
                if current_block[0].startswith('Headline:'):
                    headline = current_block[0].replace('Headline:', '').strip()
                    # The rest is the article content
                    article = '\n'.join(current_block[1:]).strip()
                    
                    headlines.append(headline)
                    articles.append(article)
                current_block = []
        else:
            current_block.append(line)
    
    # Don't forget the last block if file doesn't end with empty line
    if current_block:
        if current_block[0].startswith('Headline:'):
            headline = current_block[0].replace('Headline:', '').strip()
            article = '\n'.join(current_block[1:]).strip()
            headlines.append(headline)
            articles.append(article)
    
    return articles, headlines


def main():
    """
    Main function that reads articles and displays them.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Call the function to read articles and get both arrays
    articles, headlines = read_articles(args.filename)
    
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
