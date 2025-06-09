from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import streamlit as st
import pandas as pd
import subprocess
import os
import pickle
import re
import string
import nltk
import numpy as np
import ssl
import plotly.express as px
from scraper.twitter_scraper import Twitter_Scraper

# Initialize SSL context for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Force download the nltk data at the beginning
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
except Exception as e:
    print(f"Initial NLTK download error: {e}")


# Custom safe tokenization function


@st.cache_resource
def cleanup_scraped_data():
    """
    Clean up the scraped data file at the end of the app session
    """
    scraped_data_path = "./tweets/scraped_data.csv"
    try:
        if os.path.exists(scraped_data_path):
            os.remove(scraped_data_path)
            print(f"Deleted scraped data file: {scraped_data_path}")
    except Exception as e:
        print(f"Error deleting scraped data file: {e}")


def safe_tokenize(text):
    """Safely tokenize text even if punkt isn't properly loaded"""
    if not isinstance(text, str) or not text.strip():
        return []

    try:
        # Try standard tokenization first
        return nltk.word_tokenize(text)
    except LookupError:
        # Fallback to simple splitting
        return text.split()

# Download NLTK resources correctly


@st.cache_resource
def download_nltk_resources():
    try:
        # Try to directly ensure the punkt tokenizer is available
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        return True
    except LookupError:
        try:
            # If not found, try downloading again
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            return True
        except Exception as e:
            st.error(f"Error downloading NLTK resources: {str(e)}")
            return False


# Set page configuration
st.set_page_config(
    page_title="Twitter Toxicity Analyzer",
    page_icon="🐦",
    layout="wide"
)
# Download NLTK resources
nltk_ready = download_nltk_resources()
if not nltk_ready:
    st.warning(
        "NLTK resources might not be available. The app will use fallback methods for text processing.")

# Load the model and vectorizer


@st.cache_resource
def load_model_and_vectorizer():
    try:
        # Try to load the best model and vectorizer first
        if os.path.exists('model/best_model.pkl') and os.path.exists('model/best_vectorizer.pkl'):
            with open('model/best_model.pkl', 'rb') as file:
                model = pickle.load(file)

            with open('model/best_vectorizer.pkl', 'rb') as file:
                vectorizer = pickle.load(file)

            st.success("Loaded best model and vectorizer")
        # Fall back to the compatibility-named files
        elif os.path.exists('model/logistic_regression_model.pkl') and os.path.exists('model/tfidf_vectorizer.pkl'):
            with open('model/logistic_regression_model.pkl', 'rb') as file:
                model = pickle.load(file)

            with open('model/tfidf_vectorizer.pkl', 'rb') as file:
                vectorizer = pickle.load(file)

            st.success("Loaded model and vectorizer")
        else:
            st.error("Model files not found. Please train the model first.")
            return None, None

        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model or vectorizer: {str(e)}")
        return None, None

# Function to run the scraper


def run_scraper(mail, username, password):
    try:
        # Check if the scraped data file exists and delete it if it does
        scraped_data_path = "./tweets/scraped_data.csv"
        if os.path.exists(scraped_data_path):
            os.remove(scraped_data_path)
        st.info(f"Starting to scrape tweets for {username}...")

        # Create an instance of Twitter_Scraper and call scrape_tweets()
        scraper = Twitter_Scraper(mail, username, password, max_tweets=100)
        scraper.login()
        scraper.scrape_tweets(
            max_tweets=100, no_tweets_limit=False)  # Call the method
        scraper.save_to_csv()
        st.success("Scraping completed successfully!")
        return True

    except Exception as e:
        st.error(f"An error occurred while running the scraper: {str(e)}")
        return False

# Function to check if scraped data exists


def check_scraped_data():
    return os.path.exists("./tweets/scraped_data.csv")

# Function to load and process the scraped data


def load_data():
    try:
        df = pd.read_csv("./tweets/scraped_data.csv")
        return df
    except Exception as e:
        st.error(f"Error loading the scraped data: {str(e)}")
        return None

# Text preprocessing functions that match the training process


def remove_punc_dig(text):
    '''
    text : str
    This function will remove all the punctuations and digits from the "text"
    '''
    if not isinstance(text, str):
        return ""

    to_remove = string.punctuation + string.digits
    cur_text = ""
    for i in range(len(text)):
        if text[i] in to_remove:
            cur_text += " "
        else:
            cur_text += text[i].lower()
    cur_text = " ".join(cur_text.split())
    return cur_text


def remove_stop_words(text):
    '''
    text : str
    This function will remove stop words using NLTK
    '''
    if not isinstance(text, str):
        return ""

    try:
        stop_words = set(stopwords.words('english'))
        word_tokens = safe_tokenize(text)  # Use the safe tokenizer
        filtered_sentence = [w for w in word_tokens if w not in stop_words]
        return " ".join(filtered_sentence)
    except Exception as e:
        st.warning(
            f"Error in stop words removal: {str(e)}. Returning original text.")
        return text


def lemmatize_text(text):
    '''
    text : str
    Applying lemmatization for all words of "text" using NLTK
    '''
    if not isinstance(text, str):
        return ""

    try:
        lemmatizer = WordNetLemmatizer()
        word_tokens = safe_tokenize(text)  # Use the safe tokenizer
        lemmatized_sentence = [lemmatizer.lemmatize(w) for w in word_tokens]
        return " ".join(lemmatized_sentence)
    except Exception as e:
        st.warning(
            f"Error in lemmatization: {str(e)}. Returning original text.")
        return text

# Function to preprocess each tweet - matches the training process


def preprocess_tweet(text):
    try:
        # Step 1: Remove punctuation and digits
        text = remove_punc_dig(text)

        # Step 2: Remove stopwords
        text = remove_stop_words(text)

        # Step 3: Lemmatize
        text = lemmatize_text(text)

        return text
    except Exception as e:
        st.warning(
            f"Error preprocessing tweet: {str(e)}. Returning empty string.")
        return ""

# Function to preprocess the entire dataset


def preprocess_dataset(df):
    st.info("Preprocessing tweets...")

    # Check if Content column exists
    if 'Content' not in df.columns:
        st.warning(
            "'Content' column not found in the dataset. Available columns: " + ", ".join(df.columns))
        # Try to find an alternative column
        for col in ['content', 'Text', 'text', 'Tweet', 'tweet']:
            if col in df.columns:
                st.info(f"Using '{col}' column instead.")
                df['Content'] = df[col]
                break
        else:
            st.error("No suitable text column found.")
            return df

    # Create a copy of the original text before preprocessing
    df['original_tweet_text'] = df['Content'].copy()

    # Create a processed_text column using the same preprocessing steps as in training
    df['processed_text'] = df['Content'].apply(preprocess_tweet)

    st.success("Preprocessing completed!")
    return df

# Function to predict toxicity


def predict_toxicity(df, model, vectorizer):
    try:
        # Transform using the same vectorizer from training
        X = vectorizer.transform(df['processed_text'])

        # Make predictions
        predictions = model.predict(X)
        df['toxic'] = predictions

        # Add prediction probability if the model supports it
        try:
            # Check if model has predict_proba (Logistic Regression and Naive Bayes do, SVM doesn't)
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)
                df['toxic_probability'] = [prob[1] for prob in probs]
            # For SVM, use decision_function instead which returns distance from hyperplane
            elif hasattr(model, 'decision_function'):
                decisions = model.decision_function(X)
                # Normalize to a 0-1 range (not true probabilities but useful for visualization)
                df['toxic_probability'] = 1 / (1 + np.exp(-decisions))
        except Exception as e:
            st.warning(
                f"Could not calculate prediction probabilities: {str(e)}")

        return df
    except Exception as e:
        st.error(f"Error predicting toxicity: {str(e)}")
        return df

#############################
# VISUALIZATION FUNCTIONS
#############################

# Create toxic vs non-toxic pie chart


def create_toxicity_pie_chart(df):
    try:
        # Count toxic and non-toxic tweets
        toxic_count = int(df['toxic'].sum())
        non_toxic_count = len(df) - toxic_count

        # Create pie chart using plotly
        labels = ['Toxic', 'Non-Toxic']
        values = [toxic_count, non_toxic_count]
        colors = ['#FF6B6B', '#4ECDC4']

        fig = px.pie(
            values=values,
            names=labels,
            title='Tweet Toxicity Distribution',
            color_discrete_sequence=colors,
            hole=0.4
        )
        fig.update_traces(textinfo='percent+label', pull=[0.1, 0])
        fig.update_layout(legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
        return fig
    except Exception as e:
        st.error(f"Error creating toxicity pie chart: {str(e)}")
        return None

# Create verified vs non-verified pie chart


def create_verification_pie_chart(df):
    try:
        # Check if Verified column exists
        if 'Verified' not in df.columns:
            return None

        # Convert to boolean if needed
        if df['Verified'].dtype != bool:
            df['Verified'] = df['Verified'].fillna(False)
            # Try to convert strings like 'True' to boolean
            if df['Verified'].dtype == 'object':
                df['Verified'] = df['Verified'].map(
                    {'True': True, 'False': False, True: True, False: False})

        # Count verified and non-verified tweets
        verified_count = int(df['Verified'].sum())
        non_verified_count = len(df) - verified_count

        # Create pie chart
        labels = ['Verified', 'Non-Verified']
        values = [verified_count, non_verified_count]
        colors = ['#5DA5DA', '#FAA43A']

        fig = px.pie(
            values=values,
            names=labels,
            title='Account Verification Status',
            color_discrete_sequence=colors,
            hole=0.4
        )
        fig.update_traces(textinfo='percent+label', pull=[0.1, 0])
        fig.update_layout(legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
        return fig
    except Exception as e:
        st.error(f"Error creating verification pie chart: {str(e)}")
        return None

# Create engagement metrics bar chart


def create_engagement_chart(df):
    try:
        # Check if these columns exist
        required_cols = ['Likes', 'Retweets', 'Comments']
        if not all(col in df.columns for col in required_cols):
            return None

        # Convert to numeric if needed
        for col in required_cols:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Calculate average engagement metrics by toxicity
        toxic_df = df[df['toxic'] == 1]
        non_toxic_df = df[df['toxic'] == 0]

        toxic_metrics = {
            'Likes': toxic_df['Likes'].mean(),
            'Retweets': toxic_df['Retweets'].mean(),
            'Comments': toxic_df['Comments'].mean()
        }

        non_toxic_metrics = {
            'Likes': non_toxic_df['Likes'].mean(),
            'Retweets': non_toxic_df['Retweets'].mean(),
            'Comments': non_toxic_df['Comments'].mean()
        }

        # Create DataFrame for plotting
        chart_data = pd.DataFrame({
            'Metric': ['Likes', 'Retweets', 'Comments', 'Likes', 'Retweets', 'Comments'],
            'Value': [
                toxic_metrics['Likes'], toxic_metrics['Retweets'], toxic_metrics['Comments'],
                non_toxic_metrics['Likes'], non_toxic_metrics['Retweets'], non_toxic_metrics['Comments']
            ],
            'Toxicity': ['Toxic', 'Toxic', 'Toxic', 'Non-Toxic', 'Non-Toxic', 'Non-Toxic']
        })

        # Create grouped bar chart
        fig = px.bar(
            chart_data,
            x='Metric',
            y='Value',
            color='Toxicity',
            title='Average Engagement by Toxicity',
            barmode='group',
            color_discrete_sequence=['#FF6B6B', '#4ECDC4']
        )

        fig.update_layout(
            xaxis_title='Engagement Type',
            yaxis_title='Average Count',
            legend=dict(orientation="h", yanchor="bottom",
                        y=1.02, xanchor="center", x=0.5)
        )

        return fig
    except Exception as e:
        st.error(f"Error creating engagement chart: {str(e)}")
        return None

# Create timeline of toxic tweets


def create_toxicity_timeline(df):
    try:
        # Check if Timestamp column exists
        if 'Timestamp' not in df.columns:
            return None

        # Convert Timestamp to datetime if it's not already
        if df['Timestamp'].dtype != 'datetime64[ns]':
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

        # Drop rows with invalid timestamps
        df = df.dropna(subset=['Timestamp'])

        if len(df) == 0:
            return None

        # Group by date and toxicity
        df['Date'] = df['Timestamp'].dt.date
        timeline_data = df.groupby(
            ['Date', 'toxic']).size().reset_index(name='Count')

        # Convert to wide format
        timeline_wide = timeline_data.pivot(
            index='Date', columns='toxic', values='Count').reset_index()
        timeline_wide = timeline_wide.fillna(0)

        # Rename columns
        if 0 in timeline_wide.columns:
            timeline_wide = timeline_wide.rename(columns={0: 'Non-Toxic'})
        else:
            timeline_wide['Non-Toxic'] = 0

        if 1 in timeline_wide.columns:
            timeline_wide = timeline_wide.rename(columns={1: 'Toxic'})
        else:
            timeline_wide['Toxic'] = 0

        # Create line chart
        fig = px.line(
            timeline_wide,
            x='Date',
            y=['Toxic', 'Non-Toxic'],
            title='Timeline of Tweet Toxicity',
            markers=True,
            labels={'value': 'Number of Tweets', 'variable': 'Type'}
        )

        # Update colors
        fig.update_traces(line_color='#FF6B6B', selector=dict(name='Toxic'))
        fig.update_traces(line_color='#4ECDC4',
                          selector=dict(name='Non-Toxic'))

        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Number of Tweets',
            legend=dict(orientation="h", yanchor="bottom",
                        y=1.02, xanchor="center", x=0.5)
        )

        return fig
    except Exception as e:
        st.error(f"Error creating toxicity timeline: {str(e)}")
        return None

# Create word cloud for toxic tweets


def create_word_frequency_chart(df, top_n=10):
    try:
        # Check if processed_text column exists
        if 'processed_text' not in df.columns:
            return None

        # Split toxic and non-toxic tweets
        toxic_df = df[df['toxic'] == 1]

        if len(toxic_df) == 0:
            return None

        # Get all words from toxic tweets
        all_words = ' '.join(toxic_df['processed_text']).split()

        # Count word frequencies
        word_freq = {}
        for word in all_words:
            if len(word) > 2:  # Only count words with more than 2 characters
                word_freq[word] = word_freq.get(word, 0) + 1

        # Convert to DataFrame
        word_df = pd.DataFrame(list(word_freq.items()),
                               columns=['Word', 'Frequency'])
        word_df = word_df.sort_values('Frequency', ascending=False).head(top_n)

        # Create horizontal bar chart
        fig = px.bar(
            word_df,
            x='Frequency',
            y='Word',
            title=f'Top {top_n} Words in Toxic Tweets',
            orientation='h',
            color='Frequency',
            color_continuous_scale='Reds'
        )

        fig.update_layout(
            xaxis_title='Frequency',
            yaxis_title='Word',
            yaxis={'categoryorder': 'total ascending'}
        )

        return fig
    except Exception as e:
        st.error(f"Error creating word frequency chart: {str(e)}")
        return None

# Create toxicity by verification status chart


def create_toxicity_by_verification(df):
    try:
        # Check if required columns exist
        if 'Verified' not in df.columns or 'toxic' not in df.columns:
            return None

        # Make sure Verified is properly formatted
        df['Verified'] = df['Verified'].fillna(False)
        if df['Verified'].dtype == 'object':
            df['Verified'] = df['Verified'].map(
                {'True': True, 'False': False, True: True, False: False})

        # Group by verification status and toxicity
        verification_toxicity = df.groupby(
            ['Verified', 'toxic']).size().reset_index(name='Count')

        # Create grouped bar chart
        fig = px.bar(
            verification_toxicity,
            x='Verified',
            y='Count',
            color='toxic',
            barmode='group',
            title='Toxicity by Verification Status',
            labels={'toxic': 'Toxicity', 'Verified': 'Verified Account'},
            color_discrete_map={0: '#4ECDC4', 1: '#FF6B6B'}
        )

        # Update category names
        fig.update_xaxes(
            ticktext=['Non-Verified', 'Verified'],
            tickvals=[False, True]
        )

        # Update legend labels
        fig.update_traces(name='Non-Toxic', selector=dict(name='0'))
        fig.update_traces(name='Toxic', selector=dict(name='1'))

        fig.update_layout(
            xaxis_title='Account Verification Status',
            yaxis_title='Number of Tweets',
            legend=dict(title='Tweet Type', orientation="h",
                        yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )

        return fig
    except Exception as e:
        st.error(f"Error creating toxicity by verification chart: {str(e)}")
        return None

# Create hashtag analysis chart


def create_hashtag_analysis(df, top_n=8):
    try:
        # Check if Tags column exists
        if 'Tags' not in df.columns:
            return None

        # Extract and count hashtags
        all_tags = []
        for tags in df['Tags'].dropna():
            if isinstance(tags, str):
                # Extract hashtags from the Tags field
                tag_list = [tag.strip()
                            for tag in tags.split(',') if tag.strip()]
                all_tags.extend(tag_list)

        if not all_tags:
            return None

        # Count frequencies
        tag_counts = {}
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        # Convert to DataFrame
        tag_df = pd.DataFrame(list(tag_counts.items()),
                              columns=['Hashtag', 'Count'])
        tag_df = tag_df.sort_values('Count', ascending=False).head(top_n)

        # Create chart
        fig = px.bar(
            tag_df,
            x='Hashtag',
            y='Count',
            title=f'Top {top_n} Hashtags',
            color='Count',
            color_continuous_scale='Viridis'
        )

        fig.update_layout(
            xaxis_title='Hashtag',
            yaxis_title='Frequency',
            xaxis={'categoryorder': 'total descending'}
        )

        return fig
    except Exception as e:
        st.error(f"Error creating hashtag analysis: {str(e)}")
        return None

# Create toxicity probability distribution


def create_toxicity_probability_chart(df):
    try:
        # Check if toxic_probability column exists
        if 'toxic_probability' not in df.columns:
            return None

        # Create histogram
        fig = px.histogram(
            df,
            x='toxic_probability',
            nbins=20,
            title='Toxicity Probability Distribution',
            color_discrete_sequence=['#7209B7']
        )

        fig.update_layout(
            xaxis_title='Toxicity Probability',
            yaxis_title='Number of Tweets',
            bargap=0.05
        )

        # Add a vertical line at 0.5 threshold
        fig.add_vline(x=0.5, line_dash="dash", line_color="red")

        return fig
    except Exception as e:
        st.error(f"Error creating toxicity probability chart: {str(e)}")
        return None

# Create mentions analysis


def create_mentions_analysis(df, top_n=8):
    try:
        # Check if Mentions column exists
        if 'Mentions' not in df.columns:
            return None

        # Extract and count mentions
        all_mentions = []
        for mentions in df['Mentions'].dropna():
            if isinstance(mentions, str):
                # Extract mentions from the Mentions field
                mention_list = [mention.strip()
                                for mention in mentions.split(',') if mention.strip()]
                all_mentions.extend(mention_list)

        if not all_mentions:
            return None

        # Count frequencies
        mention_counts = {}
        for mention in all_mentions:
            mention_counts[mention] = mention_counts.get(mention, 0) + 1

        # Convert to DataFrame
        mention_df = pd.DataFrame(
            list(mention_counts.items()), columns=['Mention', 'Count'])
        mention_df = mention_df.sort_values(
            'Count', ascending=False).head(top_n)

        # Create chart
        fig = px.bar(
            mention_df,
            x='Mention',
            y='Count',
            title=f'Top {top_n} Mentioned Accounts',
            color='Count',
            color_continuous_scale='Blues'
        )

        fig.update_layout(
            xaxis_title='Mentioned Account',
            yaxis_title='Frequency',
            xaxis={'categoryorder': 'total descending'}
        )

        return fig
    except Exception as e:
        st.error(f"Error creating mentions analysis: {str(e)}")
        return None


# Main app layout
st.title("Twitter Toxicity Analyzer")
st.markdown("Analyze toxicity in tweets from any Twitter account")
# Sidebar for user input
with st.sidebar:
    st.header("Twitter Account")
    mail = st.text_input("Twitter Email")
    username = st.text_input("Twitter Username", placeholder="@elonmusk")
    password = st.text_input("Twitter Password", type="password")
    scrape_button = st.button("Scrape Tweets")

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "This app scrapes tweets, preprocesses them, and analyzes them for toxicity using a machine learning model.")

    # Advanced settings
    st.markdown("---")
    st.markdown("### Advanced Settings")
    show_preprocessing = st.checkbox("Show preprocessing details", value=False)
    show_debug_info = st.checkbox("Show debug information", value=False)

    # Visualization options
    st.markdown("---")
    st.markdown("### Visualization Options")
    visualization_options = st.multiselect(
        "Select visualizations to display",
        options=[
            "Toxicity Distribution",
            "Verification Status",
            "Engagement by Toxicity",
            "Toxicity Timeline",
            "Word Frequency",
            "Toxicity by Verification",
            "Hashtag Analysis",
            "Toxicity Probability",
            "Mentions Analysis"
        ],
        default=[
            "Toxicity Distribution",
            "Verification Status",
            "Engagement by Toxicity",
            "Toxicity Timeline"
        ]
    )

# Main content area
if scrape_button:
    if mail and username and password:
        success = run_scraper(mail, username, password)
        if success:
            st.rerun()
    else:
        st.warning("Please enter mail, username and password")

# Load model and vectorizer
model, vectorizer = load_model_and_vectorizer()

# Check if scraped data exists
if check_scraped_data():
    # Load the data
    data = load_data()

    if data is not None:
        # Show debug info if enabled
        if show_debug_info:
            st.header("Debug Information")
            st.write("Raw Data Shape:", data.shape)
            st.write("Columns:", list(data.columns))
            st.write("Data Types:", data.dtypes)
            st.write("Sample Data:")
            st.dataframe(data.head(5))

        # Preprocess the data
        preprocessed_data = preprocess_dataset(data)

        # Show preprocessing details if enabled
        if show_preprocessing and 'original_tweet_text' in preprocessed_data.columns and 'processed_text' in preprocessed_data.columns:
            st.header("Preprocessing Details")
            st.markdown("Here's how the preprocessing transforms tweet text:")

            # More visual side-by-side comparison
            sample_count = min(5, len(preprocessed_data))
            for i in range(sample_count):
                st.markdown(f"### Sample {i+1}")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Original Text:**")
                    st.text(preprocessed_data['original_tweet_text'].iloc[i])

                with col2:
                    st.markdown("**Preprocessed Text:**")
                    st.text(preprocessed_data['processed_text'].iloc[i])

            st.markdown("---")

        # Predict toxicity
        if model is not None and vectorizer is not None:
            results = predict_toxicity(preprocessed_data, model, vectorizer)

            # Display summary
            st.header("Analysis Results")

            total_tweets = len(results)
            toxic_tweets = results['toxic'].sum()
            toxic_percent = (toxic_tweets / total_tweets) * \
                100 if total_tweets > 0 else 0
            non_toxic_count = total_tweets - toxic_tweets

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Tweets Analyzed", f"{total_tweets}")
            col2.metric("Toxic Tweets", f"{int(toxic_tweets)}")
            col3.metric("Non-Toxic Tweets", f"{int(non_toxic_count)}")
            col4.metric("Toxicity Percentage", f"{toxic_percent:.1f}%")

            # Add filters section
            st.header("Filters")
            col1, col2 = st.columns(2)

            with col1:
                toxicity_filter = st.multiselect(
                    "Filter by Toxicity",
                    options=["Toxic", "Non-Toxic"],
                    default=["Toxic", "Non-Toxic"]
                )

            with col2:
                if 'Timestamp' in results.columns:
                    try:
                        results['Timestamp'] = pd.to_datetime(
                            results['Timestamp'])
                        min_date = results['Timestamp'].min().date()
                        max_date = results['Timestamp'].max().date()
                        date_range = st.date_input(
                            "Date Range",
                            value=(min_date, max_date),
                            min_value=min_date,
                            max_value=max_date
                        )
                    except Exception as e:
                        if show_debug_info:
                            st.warning(f"Could not parse timestamp: {e}")

            # Apply filters
            filtered_data = results.copy()

            # Filter by toxicity
            filter_conditions = []
            if "Toxic" in toxicity_filter:
                filter_conditions.append(filtered_data['toxic'] == 1)
            if "Non-Toxic" in toxicity_filter:
                filter_conditions.append(filtered_data['toxic'] == 0)

            if filter_conditions:
                combined_filter = filter_conditions[0]
                for condition in filter_conditions[1:]:
                    combined_filter = combined_filter | condition
                filtered_data = filtered_data[combined_filter]

            # Filter by date if available
            if 'Timestamp' in filtered_data.columns and 'date_range' in locals():
                try:
                    start_date, end_date = date_range
                    filtered_data = filtered_data[
                        (filtered_data['Timestamp'].dt.date >= start_date) &
                        (filtered_data['Timestamp'].dt.date <= end_date)
                    ]
                except Exception as e:
                    if show_debug_info:
                        st.warning(f"Date filtering error: {e}")

            # Display visualizations based on user selection
            st.header("Visualizations")

            # First row of visualizations
            if "Toxicity Distribution" in visualization_options and "Verification Status" in visualization_options:
                col1, col2 = st.columns(2)

                with col1:
                    pie_chart = create_toxicity_pie_chart(filtered_data)
                    if pie_chart:
                        st.plotly_chart(pie_chart, use_container_width=True)

                with col2:
                    verification_chart = create_verification_pie_chart(
                        filtered_data)
                    if verification_chart:
                        st.plotly_chart(verification_chart,
                                        use_container_width=True)
                    else:
                        st.info("Verification status data not available")
            elif "Toxicity Distribution" in visualization_options:
                pie_chart = create_toxicity_pie_chart(filtered_data)
                if pie_chart:
                    st.plotly_chart(pie_chart, use_container_width=True)
            elif "Verification Status" in visualization_options:
                verification_chart = create_verification_pie_chart(
                    filtered_data)
                if verification_chart:
                    st.plotly_chart(verification_chart,
                                    use_container_width=True)
                else:
                    st.info("Verification status data not available")

            # Second row of visualizations
            if "Engagement by Toxicity" in visualization_options and "Toxicity Timeline" in visualization_options:
                col1, col2 = st.columns(2)

                with col1:
                    engagement_chart = create_engagement_chart(filtered_data)
                    if engagement_chart:
                        st.plotly_chart(engagement_chart,
                                        use_container_width=True)
                    else:
                        st.info("Engagement data not available")

                with col2:
                    timeline_chart = create_toxicity_timeline(filtered_data)
                    if timeline_chart:
                        st.plotly_chart(
                            timeline_chart, use_container_width=True)
                    else:
                        st.info("Timeline data not available")
            elif "Engagement by Toxicity" in visualization_options:
                engagement_chart = create_engagement_chart(filtered_data)
                if engagement_chart:
                    st.plotly_chart(engagement_chart, use_container_width=True)
                else:
                    st.info("Engagement data not available")
            elif "Toxicity Timeline" in visualization_options:
                timeline_chart = create_toxicity_timeline(filtered_data)
                if timeline_chart:
                    st.plotly_chart(timeline_chart, use_container_width=True)
                else:
                    st.info("Timeline data not available")

            # Third row of visualizations
            if "Word Frequency" in visualization_options and "Toxicity by Verification" in visualization_options:
                col1, col2 = st.columns(2)

                with col1:
                    word_freq_chart = create_word_frequency_chart(
                        filtered_data)
                    if word_freq_chart:
                        st.plotly_chart(word_freq_chart,
                                        use_container_width=True)
                    else:
                        st.info(
                            "Not enough toxic tweets to analyze word frequency")

                with col2:
                    toxicity_by_verification = create_toxicity_by_verification(
                        filtered_data)
                    if toxicity_by_verification:
                        st.plotly_chart(toxicity_by_verification,
                                        use_container_width=True)
                    else:
                        st.info("Verification data not available")
            elif "Word Frequency" in visualization_options:
                word_freq_chart = create_word_frequency_chart(filtered_data)
                if word_freq_chart:
                    st.plotly_chart(word_freq_chart, use_container_width=True)
                else:
                    st.info("Not enough toxic tweets to analyze word frequency")
            elif "Toxicity by Verification" in visualization_options:
                toxicity_by_verification = create_toxicity_by_verification(
                    filtered_data)
                if toxicity_by_verification:
                    st.plotly_chart(toxicity_by_verification,
                                    use_container_width=True)
                else:
                    st.info("Verification data not available")

            # Fourth row of visualizations
            if "Hashtag Analysis" in visualization_options and "Toxicity Probability" in visualization_options:
                col1, col2 = st.columns(2)

                with col1:
                    hashtag_chart = create_hashtag_analysis(filtered_data)
                    if hashtag_chart:
                        st.plotly_chart(
                            hashtag_chart, use_container_width=True)
                    else:
                        st.info("Hashtag data not available")

                with col2:
                    prob_chart = create_toxicity_probability_chart(
                        filtered_data)
                    if prob_chart:
                        st.plotly_chart(prob_chart, use_container_width=True)
                    else:
                        st.info("Toxicity probability data not available")
            elif "Hashtag Analysis" in visualization_options:
                hashtag_chart = create_hashtag_analysis(filtered_data)
                if hashtag_chart:
                    st.plotly_chart(hashtag_chart, use_container_width=True)
                else:
                    st.info("Hashtag data not available")
            elif "Toxicity Probability" in visualization_options:
                prob_chart = create_toxicity_probability_chart(filtered_data)
                if prob_chart:
                    st.plotly_chart(prob_chart, use_container_width=True)
                else:
                    st.info("Toxicity probability data not available")

            # Fifth row of visualizations
            if "Mentions Analysis" in visualization_options:
                mentions_chart = create_mentions_analysis(filtered_data)
                if mentions_chart:
                    st.plotly_chart(mentions_chart, use_container_width=True)
                else:
                    st.info("Mentions data not available")

            # Display tweets in a more engaging way (from the second version)
            st.header("Analyzed Tweets")
            st.markdown(f"### Showing {len(filtered_data)} tweets")

            for i, row in filtered_data.iterrows():
                col1, col2 = st.columns([3, 1])

                with col1:
                    # Display tweet info
                    handle = row.get('Handle', 'Unknown')
                    name = row.get('Name', 'Unknown User')
                    timestamp = row.get('Timestamp', 'Unknown Date')

                    # Format timestamp if possible
                    if isinstance(timestamp, pd.Timestamp):
                        timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")

                    # Add verified badge if applicable
                    verified_text = "✓" if row.get('Verified') == True else ""

                    st.markdown(
                        f"**{name}** ({handle}) {verified_text} · {timestamp}")
                    st.markdown(row.get('original_tweet_text', row.get(
                        'Content', 'No content available')))

                    # Show mentions if available
                    mentions = row.get('Mentions', '')
                    if mentions and not pd.isna(mentions) and mentions != '':
                        st.markdown(f"**Mentions:** {mentions}")

                    # Show tweet link if available
                    tweet_link = row.get('Tweet Link', '')
                    if tweet_link and not pd.isna(tweet_link) and tweet_link != '':
                        st.markdown(f"[View Original Tweet]({tweet_link})")

                with col2:
                    # Show engagement metrics
                    likes = row.get('Likes', 0)
                    retweets = row.get('Retweets', 0)
                    comments = row.get('Comments', 0)

                    st.markdown(f"❤️ {likes} · 🔄 {retweets} · 💬 {comments}")

                    # Display toxicity badge
                    is_toxic = row['toxic']
                    if is_toxic == 1:
                        st.markdown("🚨 **Toxic**")
                        if 'toxic_probability' in row:
                            st.markdown(
                                f"Probability: {row['toxic_probability']:.2f}")
                    else:
                        st.markdown("✅ **Non-Toxic**")
                        if 'toxic_probability' in row:
                            st.markdown(
                                f"Probability: {1-row['toxic_probability']:.2f}")

                st.markdown("---")

            # Display the data table with all results
            st.header("Full Data Table")
            # Select relevant columns to display
            display_cols = ['original_tweet_text', 'toxic']
            if 'toxic_probability' in results.columns:
                display_cols.append('toxic_probability')
            if 'Timestamp' in results.columns:
                display_cols.append('Timestamp')
            if 'Likes' in results.columns:
                display_cols.append('Likes')
            if 'Retweets' in results.columns:
                display_cols.append('Retweets')
            if 'Comments' in results.columns:
                display_cols.append('Comments')
            if 'Verified' in results.columns:
                display_cols.append('Verified')
            if 'Name' in results.columns:
                display_cols.append('Name')
            if 'Handle' in results.columns:
                display_cols.append('Handle')

            # Filter by available columns
            display_cols = [
                col for col in display_cols if col in results.columns]

            # Display the dataframe
            st.dataframe(filtered_data[display_cols])
        else:
            st.error(
                "Model or vectorizer not loaded. Please make sure the model files exist.")
    else:
        st.error("Could not load the scraped data.")
else:
    st.info("No scraped data found. Please enter your Twitter credentials and click 'Scrape Tweets' to begin analysis.")

    # Display a sample of what the app will look like
    st.markdown("### Preview of Tweet Analysis")
    st.markdown(
        "Once you scrape tweets, they will be analyzed and displayed here with toxicity predictions.")
cleanup_scraped_data()
