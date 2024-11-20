# Statistical-Comparision-
This study analyzes movies from Netflix, Hulu, Prime Video, and Disney+ to evaluate age restrictions and movie quality using details like titles, age ratings, and Rotten Tomatoes scores. It explores whether Disney+ is less restrictive than Netflix and if Netflix movies are higher in quality. 
# Mount Google Drive to access the dataset
from google.colab import drive
drive.mount('/content/drive')

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# Load the dataset
data = pd.read_csv("/content/drive/MyDrive/University dortmund project/MoviesOnStreamingPlatforms.csv")

# Initial Data Exploration
print("Data Info:")
print(data.info())
print("\nData Description:")
print(data.describe())

# Check for missing values in important columns
print("\nMissing Values:")
print(data.isnull().sum())

# Filter for movies available on either Netflix or Disney+ and create a copy
netflix_disney = data[(data['Netflix'] == 1) | (data['Disney+'] == 1)].copy()

# Create a 'Platform' column to label each movie's platform
netflix_disney['Platform'] = netflix_disney.apply(lambda row: 'Netflix' if row['Netflix'] == 1 else 'Disney+', axis=1)

# Handle 'Age' column by mapping age ratings to numeric values
age_mapping = {
    'G': 0,        # General Audience
    'PG': 1,       # Parental Guidance
    'PG-13': 2,    # Parents Strongly Cautioned
    'R': 3,        # Restricted
    'NC-17': 4,    # Adults Only
    '7+': 1,       # Assuming '7+' is similar to 'PG'
    '13+': 2,      # Assuming '13+' is similar to 'PG-13'
    '16+': 3,      # Assuming '16+' is similar to 'R'
    '18+': 4       # Assuming '18+' is similar to 'NC-17'
}
netflix_disney['Age'] = netflix_disney['Age'].map(age_mapping)

# Clean 'Rotten Tomatoes' column
# Extract the first number from '98/100' format and convert to float
netflix_disney['Rotten Tomatoes'] = netflix_disney['Rotten Tomatoes'].str.extract('(\d+)').astype(float)

# Drop rows with missing values in 'Age' or 'Rotten Tomatoes'
netflix_disney.dropna(subset=['Age', 'Rotten Tomatoes'], inplace=True)

# Separate age data for Disney+ and Netflix
disney_age = netflix_disney[netflix_disney['Platform'] == 'Disney+']['Age']
netflix_age = netflix_disney[netflix_disney['Platform'] == 'Netflix']['Age']

# Check if both samples contain data
print("\nNumber of Disney+ movies with Age Data:", disney_age.shape[0])
print("Number of Netflix movies with Age Data:", netflix_age.shape[0])

# Perform T-test on Age restrictions if both samples have data
if disney_age.shape[0] > 0 and netflix_age.shape[0] > 0:
    t_stat_age, p_value_age = ttest_ind(disney_age, netflix_age, alternative='less')
    print("\nT-test result for Age Restriction (t-statistic: {}, p-value: {})".format(t_stat_age, p_value_age))

    # Interpret T-test result for Age Restrictions
    if p_value_age < 0.05:
        print("Reject the null hypothesis: The average age restriction for movies on Disney+ is lower than on Netflix.")
    else:
        print("Fail to reject the null hypothesis: No significant difference in age restriction between Disney+ and Netflix.")
else:
    print("Insufficient data for one or both platforms to perform T-test on Age restrictions.")

# Plot Age Distribution by Platform
sns.countplot(data=netflix_disney, x='Age', hue='Platform')
plt.title('Age Distribution by Platform')
plt.show()

# Rotten Tomatoes Score Comparison - Boxplot
sns.boxplot(data=netflix_disney, x='Platform', y='Rotten Tomatoes')
plt.title('Rotten Tomatoes Scores by Platform')
plt.show()

# Check Rotten Tomatoes data availability for both platforms
disney_rt = netflix_disney[(netflix_disney['Platform'] == 'Disney+') & (netflix_disney['Rotten Tomatoes'].notna())]
netflix_rt = netflix_disney[(netflix_disney['Platform'] == 'Netflix') & (netflix_disney['Rotten Tomatoes'].notna())]

print(f"\nNumber of Disney+ movies with Rotten Tomatoes scores: {len(disney_rt)}")
print(f"Number of Netflix movies with Rotten Tomatoes scores: {len(netflix_rt)}")

# Display first few rows of data with Rotten Tomatoes scores for both platforms
print("\nFirst few rows of Disney+ movies with Rotten Tomatoes scores:")
print(disney_rt[['Title', 'Rotten Tomatoes']].head())

print("\nFirst few rows of Netflix movies with Rotten Tomatoes scores:")
print(netflix_rt[['Title', 'Rotten Tomatoes']].head())

# Perform a t-test on Rotten Tomatoes scores if both platforms have sufficient data
if len(disney_rt) > 0 and len(netflix_rt) > 0:
    t_stat_rt, p_value_rt = ttest_ind(disney_rt['Rotten Tomatoes'], netflix_rt['Rotten Tomatoes'], alternative='greater')
    print("\nT-test result for Rotten Tomatoes Scores (t-statistic: {}, p-value: {})".format(t_stat_rt, p_value_rt))

    # Interpretation of T-test for Rotten Tomatoes scores
    if p_value_rt < 0.05:
        print("Reject the null hypothesis: Netflix movies have higher Rotten Tomatoes scores than Disney+ movies.")
    else:
        print("Fail to reject the null hypothesis: No significant difference in Rotten Tomatoes scores between Netflix and Disney+.")
else:
    print("Error: Insufficient Rotten Tomatoes scores for one or both platforms to perform T-test.")

