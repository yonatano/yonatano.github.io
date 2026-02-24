import markdown
import os
import re
from datetime import datetime
from dateutil import parser
from jinja2 import Environment, FileSystemLoader

# Configure Jinja2 environment
env = Environment(loader=FileSystemLoader('templates'))

import re

def title_to_filename(title, max_length=50):
    """
    Convert a blog post title into a valid filename.
    
    Args:
    title (str): The title of the blog post.
    max_length (int): The maximum length of the filename.
    
    Returns:
    str: A valid filename derived from the blog post title.
    """
    # Remove all non-word characters (everything except numbers and letters)
    title = re.sub(r'[^\w\s]', '', title)
    
    # Replace all runs of whitespace with a single dash
    title = re.sub(r'\s+', '-', title)
    
    # Convert to lowercase
    title = title.lower()
    
    # Truncate to the maximum length, ensuring we don't cut off in the middle of a word
    if len(title) > max_length:
        # Find the last dash within the max length
        last_dash = title.rfind('-', 0, max_length)
        if last_dash > 0:
            # If found, truncate up to the last dash
            title = title[:last_dash]
        else:
            # If no dash found, truncate to max length
            title = title[:max_length]
    
    # Add the HTML file extension
    title = title + '.html'
    
    return title

def parse_metadata(text):
    """Parse metadata from the given text and return a metadata dictionary and the markdown content."""
    lines = text.split('\n')
    metadata = {}
    content_start = 0

    for i, line in enumerate(lines):
        match = re.match(r'^([A-Za-z]+):\s*(.*)$', line)
        if match:
            metadata[match.group(1).lower()] = match.group(2)
            content_start = i + 1
        else:
            break  # No more metadata entries

    content = '\n'.join(lines[content_start:])
    return metadata, content

def main():
    # List all markdown files in md/ directory
    md_files = [f for f in os.listdir('md') if f.endswith('.md')]
    posts = []

    # Read each markdown file and parse metadata
    for md_file in md_files:
        with open(os.path.join('md', md_file), 'r', encoding='utf-8') as f:
            markdown_content = f.read()
            # Use raw string literals for LaTeX or double the backslashes
            markdown_content = markdown_content.replace("\\", "\\\\")  
            metadata, content = parse_metadata(markdown_content)

        # Convert markdown to HTML
        md = markdown.Markdown(
            extensions=['extra', 'codehilite', 'toc'],
            extension_configs={
                'codehilite': {'css_class': 'language-python'},
                'toc': {'toc_depth': 2}
            }
        )
        html_content = md.convert(content)
        toc = md.toc  # empty string '' when no headings present

        # Generate post HTML file
        post_template = env.get_template('post.html')
        post_html = post_template.render(
            title=metadata['title'],
            time=metadata['time'],
            tags=metadata['tags'],
            content=html_content,
            toc=toc,
            stylesheet='style.css'
        )

        post_filename = os.path.join('posts', title_to_filename(metadata['title']))
        with open(post_filename, 'w', encoding='utf-8') as f:
            f.write(post_html)

        # Skip if draft.
        if metadata['publish'] != 'True':
            continue

        # Add a date key to the metadata dictionary.
        metadata['date'] = parser.parse(metadata['time'])

        # Collect data for blog listing
        preview = ' '.join(content.split('\n')[0:2])  # First two sentences as preview
        posts.append({
            'title': metadata['title'],
            'time': metadata['time'],
            'tags': metadata['tags'],
            'url': post_filename,
            'preview': preview,
            'date': metadata['date']
        })

    # Sort posts by the 'date' key in descending order
    posts.sort(key=lambda x: x['date'], reverse=True)

    # Calculate days since last post
    current_time = datetime.now()
    days_since_last_post = (current_time - posts[0]['date']).days if posts else 0

    # Generate blog listing HTML file
    blog_template = env.get_template('blog.html')
    blog_html = blog_template.render(posts=posts, days_since_last_post=days_since_last_post)
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(blog_html)

    print('Blog generated successfully.')

if __name__ == '__main__':
    main()
