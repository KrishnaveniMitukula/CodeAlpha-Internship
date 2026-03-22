import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# Download required NLTK data
print("📦 Downloading NLTK data...")
nltk.download("stopwords",    quiet=True)
nltk.download("punkt",        quiet=True)
nltk.download("wordnet",      quiet=True)
nltk.download("punkt_tab",    quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
print("✅ NLTK data ready!")

sns.set_theme(style="whitegrid")
np.random.seed(42)

# ─────────────────────────────────────
# STEP 1: Create Amazon Reviews Dataset
# ─────────────────────────────────────
print("\n" + "="*55)
print("📦 STEP 1: Creating Amazon Reviews Dataset")
print("="*55)

reviews_data = [
    # Positive Reviews
    ("This product is absolutely amazing! Best purchase ever.",        5, "Electronics"),
    ("Fantastic quality, exceeded my expectations completely!",        5, "Electronics"),
    ("Superb product, works perfectly. Highly recommend to everyone!", 5, "Electronics"),
    ("Excellent value for money. Very happy with this purchase.",      5, "Electronics"),
    ("Outstanding performance, delivery was fast and packaging great!",5, "Electronics"),
    ("Love this product! It works exactly as described. Very happy.",  5, "Clothing"),
    ("Perfect fit and great quality material. Will buy again soon!",   5, "Clothing"),
    ("Beautiful design and very comfortable to wear daily.",           4, "Clothing"),
    ("Great product at an affordable price. Totally worth it!",        4, "Clothing"),
    ("Very satisfied with purchase. Quality is better than expected.", 4, "Clothing"),
    ("Delicious taste! Best food product I have ever ordered online.", 5, "Food"),
    ("Amazing flavor and very fresh. Will definitely order again!",    5, "Food"),
    ("Great taste and good quality packaging. Very impressed!",        4, "Food"),
    ("Wonderful product, kids absolutely love it every single day.",   5, "Food"),
    ("Tasty and healthy. Perfect for my daily breakfast routine.",     4, "Food"),

    # Neutral Reviews
    ("Product is okay. Nothing special but does the job fine.",        3, "Electronics"),
    ("Average quality, works as expected. Delivery was on time.",      3, "Electronics"),
    ("It is decent. Not the best but not the worst either.",           3, "Electronics"),
    ("Mediocre product. Some features work well, others do not.",      3, "Electronics"),
    ("Acceptable quality for the price. Could be better overall.",     3, "Clothing"),
    ("Fits okay. Color is slightly different from picture shown.",     3, "Clothing"),
    ("Average material quality. Neither good nor bad experience.",     3, "Clothing"),
    ("Product is fine. Does what it says nothing more nothing less.",  3, "Food"),
    ("Taste is average. Not as good as expected from description.",    3, "Food"),
    ("Okay product. Packaging could be improved for better safety.",   3, "Food"),

    # Negative Reviews
    ("Terrible product! Stopped working after just two days use.",     1, "Electronics"),
    ("Very disappointed. Quality is extremely poor for this price.",   1, "Electronics"),
    ("Worst purchase ever made. Complete waste of money spent.",       1, "Electronics"),
    ("Product broke immediately. Very bad quality and poor build.",    2, "Electronics"),
    ("Horrible experience. Customer service was also very rude.",      1, "Electronics"),
    ("Does not fit at all. Size chart is completely wrong here.",      1, "Clothing"),
    ("Very poor quality material. Started falling apart quickly.",     2, "Clothing"),
    ("Extremely disappointed with this product. Not worth money.",     1, "Clothing"),
    ("Bad stitching and wrong color. Will never buy from here again.", 2, "Clothing"),
    ("Terrible quality clothing. Looks nothing like the picture.",     1, "Clothing"),
    ("Awful taste and smell was very bad. Totally inedible product.",  1, "Food"),
    ("Food arrived damaged and expired. Very dangerous product.",      1, "Food"),
    ("Disgusting product. Made me feel sick after eating it.",         1, "Food"),
    ("Worst food ever ordered. Taste is absolutely horrible.",         2, "Food"),
    ("Very bad quality food. Packaging was broken and leaking badly.", 2, "Food"),
]

df = pd.DataFrame(reviews_data,
                  columns=["Review", "Rating", "Category"])
print(f"✅ Dataset created: {len(df)} reviews")
print(f"   Categories: {df['Category'].unique().tolist()}")
print(f"   Rating range: {df['Rating'].min()} - {df['Rating'].max()}")

# ─────────────────────────────────────
# STEP 2: Sentiment Analysis
# ─────────────────────────────────────
print("\n" + "="*55)
print("🔍 STEP 2: Analyzing Sentiments")
print("="*55)

def analyze_sentiment(text):
    """Analyze sentiment using TextBlob"""
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity

    if polarity > 0.1:
        sentiment = "Positive"
    elif polarity < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return polarity, sentiment

# Apply sentiment analysis
df["Polarity"]  = df["Review"].apply(
    lambda x: analyze_sentiment(x)[0])
df["Sentiment"] = df["Review"].apply(
    lambda x: analyze_sentiment(x)[1])

# Subjectivity
df["Subjectivity"] = df["Review"].apply(
    lambda x: TextBlob(x).sentiment.subjectivity)

print(f"\n  Sentiment Distribution:")
sent_counts = df["Sentiment"].value_counts()
for sent, count in sent_counts.items():
    pct = count/len(df)*100
    print(f"   {sent:<12}: {count:>3} reviews ({pct:.1f}%)")

print(f"\n  Average Polarity by Category:")
cat_pol = df.groupby("Category")["Polarity"].mean()
for cat, pol in cat_pol.items():
    mood = "😊 Positive" if pol > 0 else "😞 Negative"
    print(f"   {cat:<20}: {pol:>6.3f} ({mood})")

# ─────────────────────────────────────
# STEP 3: Text Processing
# ─────────────────────────────────────
print("\n" + "="*55)
print("🧹 STEP 3: Text Processing")
print("="*55)

stop_words  = set(stopwords.words("english"))
lemmatizer  = WordNetLemmatizer()

def process_text(text):
    """Clean and process text"""
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(t)
              for t in tokens
              if t.isalpha() and t not in stop_words]
    return tokens

df["Tokens"] = df["Review"].apply(process_text)

# Most common words by sentiment
for sentiment in ["Positive", "Negative", "Neutral"]:
    subset = df[df["Sentiment"] == sentiment]["Tokens"]
    all_words = [w for tokens in subset for w in tokens]
    common = Counter(all_words).most_common(5)
    words_str = ", ".join([f"{w}({c})" for w,c in common])
    print(f"  {sentiment} top words: {words_str}")

# ─────────────────────────────────────
# STEP 4: Visualizations
# ─────────────────────────────────────
print("\n🎨 STEP 4: Creating Visualizations...")

COLORS = {
    "Positive": "#2ecc71",
    "Neutral":  "#f39c12",
    "Negative": "#e74c3c"
}

fig = plt.figure(figsize=(20, 14), facecolor="#f8f9fa")
gs  = gridspec.GridSpec(2, 3, figure=fig,
                        hspace=0.4, wspace=0.35,
                        top=0.90, bottom=0.08,
                        left=0.06, right=0.97)

fig.suptitle("Amazon Reviews — Sentiment Analysis Dashboard\n"
             "CodeAlpha Data Analytics Internship — Task 4",
             fontsize=16, fontweight="bold", y=0.97)

# ── Plot 1: Sentiment Distribution Pie ──
ax1 = fig.add_subplot(gs[0, 0])
sizes  = sent_counts.values
labels = sent_counts.index
colors = [COLORS[s] for s in labels]
wedges, texts, autotexts = ax1.pie(
    sizes, labels=labels, autopct="%1.1f%%",
    colors=colors, startangle=140,
    wedgeprops={"edgecolor": "white", "linewidth": 2})
for t in autotexts:
    t.set_fontweight("bold")
ax1.set_title("Overall Sentiment\nDistribution",
              fontweight="bold", fontsize=12)

# ── Plot 2: Sentiment by Category ──
ax2 = fig.add_subplot(gs[0, 1])
cat_sent = df.groupby(["Category","Sentiment"])\
             .size().unstack(fill_value=0)
cat_sent_pct = cat_sent.div(cat_sent.sum(axis=1), axis=0) * 100
bottom_vals = np.zeros(len(cat_sent_pct))
for sentiment in ["Positive", "Neutral", "Negative"]:
    if sentiment in cat_sent_pct.columns:
        vals = cat_sent_pct[sentiment].values
        ax2.bar(cat_sent_pct.index, vals,
                bottom=bottom_vals,
                color=COLORS[sentiment],
                label=sentiment,
                edgecolor="white")
        bottom_vals += vals
ax2.set_title("Sentiment by Category",
              fontweight="bold", fontsize=12)
ax2.set_ylabel("Percentage (%)")
ax2.legend(fontsize=8)
ax2.set_xticklabels(cat_sent_pct.index, rotation=15)

# ── Plot 3: Polarity Distribution ──
ax3 = fig.add_subplot(gs[0, 2])
pos_data = df[df["Sentiment"]=="Positive"]["Polarity"]
neg_data = df[df["Sentiment"]=="Negative"]["Polarity"]
neu_data = df[df["Sentiment"]=="Neutral"]["Polarity"]
ax3.hist(pos_data, bins=10, alpha=0.7,
         color=COLORS["Positive"], label="Positive")
ax3.hist(neg_data, bins=10, alpha=0.7,
         color=COLORS["Negative"], label="Negative")
ax3.hist(neu_data, bins=10, alpha=0.7,
         color=COLORS["Neutral"],  label="Neutral")
ax3.set_title("Polarity Score\nDistribution",
              fontweight="bold", fontsize=12)
ax3.set_xlabel("Polarity Score (-1 to +1)")
ax3.set_ylabel("Count")
ax3.legend(fontsize=8)
ax3.axvline(x=0, color="black", linestyle="--",
            lw=1.5, alpha=0.7)

# ── Plot 4: Rating vs Polarity ──
ax4 = fig.add_subplot(gs[1, 0])
colors_scatter = [COLORS[s] for s in df["Sentiment"]]
ax4.scatter(df["Rating"], df["Polarity"],
            c=colors_scatter, alpha=0.7,
            s=80, edgecolors="white", lw=0.5)
ax4.set_title("Rating vs Polarity Score",
              fontweight="bold", fontsize=12)
ax4.set_xlabel("Star Rating (1-5)")
ax4.set_ylabel("Polarity Score")
ax4.set_xticks([1, 2, 3, 4, 5])
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLORS["Positive"], label="Positive"),
    Patch(facecolor=COLORS["Neutral"],  label="Neutral"),
    Patch(facecolor=COLORS["Negative"], label="Negative")
]
ax4.legend(handles=legend_elements, fontsize=8)

# ── Plot 5: Average Polarity by Category ──
ax5 = fig.add_subplot(gs[1, 1])
cat_polarity = df.groupby("Category")["Polarity"].mean()\
                 .sort_values(ascending=True)
bar_colors = ["#2ecc71" if v > 0 else "#e74c3c"
              for v in cat_polarity.values]
bars = ax5.barh(cat_polarity.index, cat_polarity.values,
                color=bar_colors, edgecolor="white")
for bar in bars:
    width = bar.get_width()
    ax5.text(width + 0.01,
             bar.get_y() + bar.get_height()/2,
             f"{width:.3f}",
             va="center", fontsize=10,
             fontweight="bold")
ax5.set_title("Avg Polarity by Category",
              fontweight="bold", fontsize=12)
ax5.set_xlabel("Average Polarity Score")
ax5.axvline(x=0, color="black",
            linestyle="--", lw=1.5)

# ── Plot 6: Subjectivity vs Polarity ──
ax6 = fig.add_subplot(gs[1, 2])
for sentiment, color in COLORS.items():
    mask = df["Sentiment"] == sentiment
    ax6.scatter(df[mask]["Subjectivity"],
                df[mask]["Polarity"],
                c=color, label=sentiment,
                alpha=0.7, s=80,
                edgecolors="white", lw=0.5)
ax6.set_title("Subjectivity vs Polarity",
              fontweight="bold", fontsize=12)
ax6.set_xlabel("Subjectivity (0=Objective, 1=Subjective)")
ax6.set_ylabel("Polarity Score")
ax6.legend(fontsize=8)
ax6.axhline(y=0, color="black",
            linestyle="--", lw=1, alpha=0.5)

plt.savefig("sentiment_analysis.png",
            dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("💾 Dashboard saved as 'sentiment_analysis.png'")
plt.show()

# ─────────────────────────────────────
# STEP 5: Key Findings
# ─────────────────────────────────────
print("\n" + "="*55)
print("📝 KEY FINDINGS — Sentiment Analysis")
print("="*55)

pos_pct = len(df[df["Sentiment"]=="Positive"])/len(df)*100
neg_pct = len(df[df["Sentiment"]=="Negative"])/len(df)*100
neu_pct = len(df[df["Sentiment"]=="Neutral"])/len(df)*100
avg_pol = df["Polarity"].mean()

print(f"  1. Positive reviews : {pos_pct:.1f}% of all reviews")
print(f"  2. Negative reviews : {neg_pct:.1f}% of all reviews")
print(f"  3. Neutral reviews  : {neu_pct:.1f}% of all reviews")
print(f"  4. Overall polarity : {avg_pol:.3f} "
      f"({'Positive' if avg_pol > 0 else 'Negative'})")
print(f"  5. Best category    : "
      f"{cat_pol.idxmax()} ({cat_pol.max():.3f})")
print(f"  6. Worst category   : "
      f"{cat_pol.idxmin()} ({cat_pol.min():.3f})")
print(f"  7. High ratings (4-5★) = mostly positive sentiment")
print(f"  8. Low ratings  (1-2★) = mostly negative sentiment")
print("="*55)
print("\n✅ Task 4 Sentiment Analysis COMPLETE!")

# Save results
df[["Review","Rating","Category",
    "Sentiment","Polarity","Subjectivity"]]\
  .to_csv("sentiment_results.csv", index=False)
print("💾 Results saved as 'sentiment_results.csv'")
