import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm
plt.style.use('ggplot')
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, sentiwordnet as swn
from nltk.corpus import wordnet as wn

# Read in data
df = pd.read_csv('Reviews.csv')

# Quick EDA
plt.figure(figsize=(10, 5))
ax = df['Score'].value_counts().sort_index().plot(kind='bar', title='Count of Reviews by Stars')
ax.set_xlabel('Review Stars')
plt.savefig('review_stars_distribution.png')
plt.close()

# Function to convert NLTK POS tags to WordNet POS tags
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        # Default to noun if not matched
        return wn.NOUN

# Function to get sentiment scores using SentiWordNet
def get_sentiment_scores(text):
    # Tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove punctuation and numbers
    tokens = [token for token in tokens if token.isalpha()]
    
    # POS tagging
    tagged = nltk.pos_tag(tokens)
    
    # Initialize sentiment scores
    pos_score = 0.0
    neg_score = 0.0
    token_count = 0
    
    # Get sentiment for each word
    for word, tag in tagged:
        # Skip stopwords
        if word in stopwords.words('english'):
            continue
            
        # Convert to WordNet POS tag
        wn_tag = get_wordnet_pos(tag)
        
        # Get SentiWordNet synsets
        synsets = list(swn.senti_synsets(word, wn_tag))
        
        # Skip words not in SentiWordNet
        if not synsets:
            continue
            
        # Use first synset (most common sense)
        synset = synsets[0]
        
        # Add scores
        pos_score += synset.pos_score()
        neg_score += synset.neg_score()
        token_count += 1
    
    # Calculate average scores
    if token_count > 0:
        pos_score = pos_score / token_count
        neg_score = neg_score / token_count
    else:
        pos_score = 0.0
        neg_score = 0.0
    
    # Calculate compound and neutral scores
    compound = pos_score - neg_score
    neu_score = 1.0 - (pos_score + neg_score)
    
    return {
        'positive': pos_score,
        'negative': neg_score,
        'neutral': neu_score,
        'compound': compound
    }

# Test on an example
example = df['Text'][50] if len(df) > 50 else df['Text'][0]
print("Example text:")
print(example)

sentiment = get_sentiment_scores(example)
print("\nSentiment scores:")
for key, value in sentiment.items():
    print(f"{key}: {value:.4f}")

# Apply sentiment analysis to all reviews
print("\nCalculating sentiment for all reviews...")
results = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    scores = get_sentiment_scores(text)
    
    result = {
        'Id': row['Id'],
        'positive': scores['positive'],
        'negative': scores['negative'],
        'neutral': scores['neutral'],
        'compound': scores['compound'],
        'Score': row['Score']
    }
    results.append(result)

# Create results dataframe
results_df = pd.DataFrame(results)

# Merge with original data if needed
sentiment_df = results_df.merge(df[['Id', 'Text']], on='Id', how='left')

# Save results
sentiment_df.to_csv('nltk_sentiment_results.csv', index=False)

# Plot results
plt.figure(figsize=(10, 6))
sns.barplot(data=sentiment_df, x='Score', y='compound')
plt.title('Compound Sentiment Score by Review Rating')
plt.xlabel('Review Rating (stars)')
plt.ylabel('Compound Sentiment Score')
plt.savefig('nltk_sentiment_by_rating.png')
plt.close()

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
sns.barplot(data=sentiment_df, x='Score', y='positive', ax=axs[0])
sns.barplot(data=sentiment_df, x='Score', y='neutral', ax=axs[1])
sns.barplot(data=sentiment_df, x='Score', y='negative', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.savefig('nltk_sentiment_components_by_rating.png')
plt.close()

# Positive 1-star reviews
print("\nAll Positive 1-Star Reviews:")
positive_1star = sentiment_df[(sentiment_df['Score'] == 1) & (sentiment_df['compound'] > 0)]
positive_1star = positive_1star.sort_values('compound', ascending=False)

if not positive_1star.empty:
    print(f"Found {len(positive_1star)} positive 1-star reviews\n")
    for i, row in positive_1star.iterrows():
        print(f"ID: {row['Id']}")
        print(f"Compound score: {row['compound']:.4f}")
        print(f"Text: {row['Text']}")
        print("-" * 80)  
else:
    print("No positive 1-star reviews found.")

# Negative 5-star reviews
print("\nAll Negative 5-Star Reviews:")
negative_5star = sentiment_df[(sentiment_df['Score'] == 5) & (sentiment_df['compound'] < 0)]
negative_5star = negative_5star.sort_values('compound')

if not negative_5star.empty:
    print(f"Found {len(negative_5star)} negative 5-star reviews\n")
    for i, row in negative_5star.iterrows():
        print(f"ID: {row['Id']}")
        print(f"Compound score: {row['compound']:.4f}")
        print(f"Text: {row['Text']}")
        print("-" * 80)  
else:
    print("No negative 5-star reviews found.")

# Calculate correlation
correlation = sentiment_df[['Score', 'compound']].corr().iloc[0, 1]
print(f"\nCorrelation between rating and sentiment: {correlation:.4f}")

print("\nNLTK sentiment analysis complete!")