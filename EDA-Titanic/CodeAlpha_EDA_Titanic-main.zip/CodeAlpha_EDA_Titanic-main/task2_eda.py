import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────
# STEP 1: Load Dataset
# ─────────────────────────────────────
print("📦 Loading Titanic dataset...")
df = sns.load_dataset("titanic")
print(f"✅ Loaded! Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# ─────────────────────────────────────
# STEP 2: Basic Inspection
# ─────────────────────────────────────
print("\n" + "="*50)
print("🔍 BASIC INFORMATION")
print("="*50)
print(f"Columns     : {list(df.columns)}")
print(f"Total Rows  : {len(df)}")
print(f"\nData Types:\n{df.dtypes.to_string()}")

print("\nMissing Values:")
missing = df.isnull().sum()
missing = missing[missing > 0]
for col, count in missing.items():
    pct = count / len(df) * 100
    print(f"   {col:<20} {count:>4} missing ({pct:.1f}%)")

# ─────────────────────────────────────
# STEP 3: Data Cleaning
# ─────────────────────────────────────
print("\n" + "="*50)
print("🧹 DATA CLEANING")
print("="*50)

df = df.copy()

# Fill missing age with median
median_age = df["age"].median()
df["age"].fillna(median_age, inplace=True)
print(f"✅ Filled missing 'age' with median ({median_age:.1f})")

# Fill missing embarked with mode
mode_embarked = df["embarked"].mode()[0]
df["embarked"].fillna(mode_embarked, inplace=True)
print(f"✅ Filled missing 'embarked' with '{mode_embarked}'")

# Drop deck column (too many missing)
df.drop(columns=["deck"], inplace=True)
print(f"✅ Dropped 'deck' column (too many missing values)")

# Create age groups
df["age_group"] = pd.cut(df["age"],
                          bins=[0, 12, 18, 35, 60, 100],
                          labels=["Child", "Teen", "Young Adult", "Adult", "Senior"])
print(f"✅ Created age_group column")

# ─────────────────────────────────────
# STEP 4: Statistical Analysis
# ─────────────────────────────────────
print("\n" + "="*50)
print("📊 STATISTICAL ANALYSIS")
print("="*50)

total = len(df)
survived = df["survived"].sum()
print(f"\nOverall Survival Rate: {survived}/{total} = {survived/total*100:.1f}%")

print("\nSurvival by Gender:")
gender_surv = df.groupby("sex")["survived"].agg(["sum","count"])
gender_surv["rate"] = gender_surv["sum"] / gender_surv["count"] * 100
for sex, row in gender_surv.iterrows():
    print(f"   {sex.capitalize():<8}: {row['sum']:.0f}/{row['count']:.0f} = {row['rate']:.1f}%")

print("\nSurvival by Class:")
class_surv = df.groupby("pclass")["survived"].agg(["sum","count"])
class_surv["rate"] = class_surv["sum"] / class_surv["count"] * 100
for cls, row in class_surv.iterrows():
    print(f"   Class {cls}: {row['sum']:.0f}/{row['count']:.0f} = {row['rate']:.1f}%")

print("\nAge Statistics:")
print(f"   Mean Age   : {df['age'].mean():.1f}")
print(f"   Median Age : {df['age'].median():.1f}")
print(f"   Min Age    : {df['age'].min():.0f}")
print(f"   Max Age    : {df['age'].max():.0f}")

# ─────────────────────────────────────
# STEP 5: Visualizations
# ─────────────────────────────────────
print("\n🎨 Creating Visualizations...")
sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Titanic EDA — CodeAlpha Data Analytics Internship",
             fontsize=15, fontweight="bold")

# Plot 1: Survival Count
axes[0,0].bar(["Did Not Survive", "Survived"],
              df["survived"].value_counts().values,
              color=["#e74c3c", "#2ecc71"], edgecolor="white")
axes[0,0].set_title("Overall Survival Count", fontweight="bold")
axes[0,0].set_ylabel("Count")

# Plot 2: Survival by Gender
gender_plot = df.groupby(["sex","survived"]).size().unstack()
gender_plot.plot(kind="bar", ax=axes[0,1],
                 color=["#e74c3c","#2ecc71"], edgecolor="white")
axes[0,1].set_title("Survival by Gender", fontweight="bold")
axes[0,1].set_xticklabels(["Female","Male"], rotation=0)
axes[0,1].legend(["Died","Survived"], fontsize=8)

# Plot 3: Survival by Class
class_plot = df.groupby(["pclass","survived"]).size().unstack()
class_plot.plot(kind="bar", ax=axes[0,2],
                color=["#e74c3c","#2ecc71"], edgecolor="white")
axes[0,2].set_title("Survival by Class", fontweight="bold")
axes[0,2].set_xticklabels(["1st","2nd","3rd"], rotation=0)
axes[0,2].legend(["Died","Survived"], fontsize=8)

# Plot 4: Age Distribution
df[df["survived"]==1]["age"].plot(kind="hist", ax=axes[1,0],
    bins=20, alpha=0.6, color="#2ecc71", label="Survived")
df[df["survived"]==0]["age"].plot(kind="hist", ax=axes[1,0],
    bins=20, alpha=0.6, color="#e74c3c", label="Died")
axes[1,0].set_title("Age Distribution", fontweight="bold")
axes[1,0].set_xlabel("Age")
axes[1,0].legend()

# Plot 5: Fare by Class
df_fare = df[df["fare"] < 300]
sns.boxplot(data=df_fare, x="pclass", y="fare",
            ax=axes[1,1], palette=["#3498db","#9b59b6","#e67e22"])
axes[1,1].set_title("Fare by Class", fontweight="bold")
axes[1,1].set_xticklabels(["1st","2nd","3rd"])

# Plot 6: Survival Rate by Age Group
age_surv = df.groupby("age_group", observed=True)["survived"].mean() * 100
axes[1,2].bar(age_surv.index, age_surv.values,
              color=["#1abc9c","#3498db","#9b59b6","#e67e22","#e74c3c"],
              edgecolor="white")
axes[1,2].set_title("Survival Rate by Age Group", fontweight="bold")
axes[1,2].set_ylabel("Survival Rate (%)")

plt.tight_layout()
plt.savefig("eda_titanic_analysis.png", dpi=150, bbox_inches="tight")
print("💾 Chart saved as 'eda_titanic_analysis.png'")
plt.show()

# ─────────────────────────────────────
# STEP 6: Key Findings
# ─────────────────────────────────────
print("\n" + "="*50)
print("📝 KEY FINDINGS")
print("="*50)
print("1. Overall survival rate was ~38%")
print("2. Women survived at 74% vs men at only 19%")
print("3. 1st class passengers survived at 63%")
print("4. 3rd class passengers survived at only 24%")
print("5. Children had the highest survival rate")
print("6. Seniors had the lowest survival rate")
print("7. 1st class fares were much higher than 3rd class")
print("="*50)
print("\n✅ Task 2 COMPLETE!")

