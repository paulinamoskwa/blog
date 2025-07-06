
FROM ruby:3.2

WORKDIR /app

# Install required dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Jekyll and Bundler
RUN gem install jekyll bundler

# Copy Gemfile and gemspec first for better layer caching
COPY content/Gemfile content/tale.gemspec ./

# Install dependencies
RUN bundle install

# Copy the rest of the application
COPY content/ .

# Expose ports for Jekyll server and LiveReload
EXPOSE 4000 35729

# Command to run when container starts
CMD ["bundle", "exec", "jekyll", "serve", "--host", "0.0.0.0", "--watch", "--livereload"]