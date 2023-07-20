import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import re


def compare_emotions(dataframes, base='orig'):
    base_df = dataframes[base]

    # Loop over each dataframe for comparison
    for identifier, df in dataframes.items():
        if identifier == base:
            continue  # Skip the base dataframe

        print(f"\nComparison with {identifier}:")

        # Check the amount of matching emotions
        matches = (base_df['emotion'] == df['emotion']).sum()
        print(f"Number of matching emotions: {matches}")

        # Calculate the correlation
        correlation = base_df['emotion'].astype('category').cat.codes.corr(
            df['emotion'].astype('category').cat.codes
        )
        print(f"Correlation: {correlation}")

        # Emotion distribution
        print(f"Emotion distribution in {identifier}:")
        print(df['emotion'].value_counts())

        # Match/non-match comparison
        comparison = base_df['emotion'] == df['emotion']
        print("Match/non-match comparison:")
        print(comparison.value_counts(normalize=True))


# The directory containing your pickle files
directory = 'outputs/'

# Get a list of all pickle files in the directory
pickle_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]

# A dictionary to store each dataframe, using the identifier as the key
dataframes = {}

# Load each pickle file into a dataframe
for file in pickle_files:
    # Extract identifier from the filename
    identifier = re.findall(r"_(.*).pkl", file)[0]

    # Load dataframe from pickle file
    df = pd.read_pickle(os.path.join(directory, file))

    # Extract base column names (emotion and score) by removing the identifier
    df.columns = [re.sub(f'_{identifier}', '', col) for col in df.columns]

    # Check if 'emotion' column exists in the dataframe
    if 'emotion' not in df.columns:
        print(f"DataFrame from {file} does not have 'emotion' column. Skipping this file.")
        continue

    # Store dataframe in the dictionary
    dataframes[identifier] = df

    # Print the identifier and dataframe
    print(f"Identifier: {identifier}")
    print(df)

# Call the function
compare_emotions(dataframes)

# Bar plot for percentage of matching emotions:
base = 'orig'
base_df = dataframes[base]
matching_emotions = {}
for identifier, df in dataframes.items():
    if identifier == base:
        continue
    total_records = len(df)
    matches = (base_df['emotion'] == df['emotion']).sum()
    # Calculate percentage of matches
    match_percentage = (matches / total_records) * 100
    matching_emotions[identifier] = match_percentage

# Create a list of colors for the bars
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

# Create a larger figure (width, height) in inches
plt.figure(figsize=(8, 6))
plt.subplots_adjust(bottom=0.15)  # Adjust bottom margin

bars = plt.bar(matching_emotions.keys(), matching_emotions.values(), color=colors)

plt.xlabel('Dataframe')
plt.ylabel('Percentage of matching emotions')
plt.title('Matching Emotions per Dataframe')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility

# Adding the percentage values on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, round(yval, 2), ha='center', va='bottom')

plt.show()

# Bar plot for correlation:
correlations = {}
for identifier, df in dataframes.items():
    if identifier == base:
        continue
    correlation = base_df['emotion'].astype('category').cat.codes.corr(df['emotion'].astype('category').cat.codes)
    correlations[identifier] = correlation

plt.bar(correlations.keys(), correlations.values())
plt.xlabel('Dataframe')
plt.ylabel('Correlation')
plt.title('Correlation per Dataframe')
plt.show()


# Pie chart for match/non-match comparison (per dataframe):
for identifier, df in dataframes.items():
    if identifier == base:
        continue
    comparison = base_df['emotion'] == df['emotion']
    values = comparison.value_counts(normalize=True)

    plt.figure(figsize=(6, 6))
    plt.pie(values, labels=['Match', 'Non-Match'], autopct='%1.1f%%')
    plt.title(f'Match/Non-Match Comparison in {identifier} Dataframe')
    plt.show()


emotions = ['Neutral', 'Disgust', 'Contempt', 'Happiness', 'Fear', 'Anger', 'Sadness', 'Surprise']

for emotion in emotions:
    emotion_matching = {}
    for identifier, df in dataframes.items():
        if identifier == base:
            continue
        base_emotion_df = base_df[base_df['emotion'] == emotion]
        compare_emotion_df = df[df['emotion'] == emotion]
        total_records = len(base_emotion_df)
        matches = base_emotion_df['filename'].isin(compare_emotion_df['filename']).sum()
        match_percentage = (matches / total_records) * 100 if total_records > 0 else 0
        emotion_matching[identifier] = match_percentage

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    # Create a larger figure (width, height) in inches
    plt.figure(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.25)  # Adjust bottom margin

    bars = plt.bar(emotion_matching.keys(), emotion_matching.values(), color=colors)

    plt.xlabel('Dataframe')
    plt.ylabel('Percentage of matching emotions')
    plt.title(f'Matching {emotion} per Dataframe')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility

    # Adding the percentage values on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, round(yval, 2), ha='center', va='bottom')

    plt.show()


emotions = ['Neutral', 'Disgust', 'Contempt', 'Happiness', 'Fear', 'Anger', 'Sadness', 'Surprise']
base = 'orig'
base_df = dataframes[base]

for identifier, df in dataframes.items():
    if identifier == base:
        continue
    emotion_matching = {}
    for emotion in emotions:
        base_emotion_df = base_df[base_df['emotion'] == emotion]
        compare_emotion_df = df[df['emotion'] == emotion]
        total_records = len(base_emotion_df)
        matches = base_emotion_df['filename'].isin(compare_emotion_df['filename']).sum()
        match_percentage = (matches / total_records) * 100 if total_records > 0 else 0
        emotion_matching[emotion] = match_percentage

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple']

    # Create a larger figure (width, height) in inches
    plt.figure(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.18)  # Adjust bottom margin

    bars = plt.bar(emotion_matching.keys(), emotion_matching.values(), color=colors)

    plt.xlabel('Emotions')
    plt.ylabel('Percentage of matching emotions')
    plt.title(f'Matching emotions for dataframe: {identifier}')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility

    # Adding the percentage values on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, round(yval, 2), ha='center', va='bottom')

    plt.show()
    
"""
### check if the following is just a duplicate...
"""

emotions = ['Neutral', 'Disgust', 'Contempt', 'Happiness', 'Fear', 'Anger', 'Sadness', 'Surprise']
base = 'orig'
base_df = dataframes[base]

for identifier, df in dataframes.items():
    if identifier == base:
        continue
    emotion_matching = {}
    for emotion in emotions:
        base_emotion_df = base_df[base_df['emotion'] == emotion]
        compare_emotion_df = df[df['emotion'] == emotion]
        total_records = len(base_emotion_df)
        matches = base_emotion_df['filename'].isin(compare_emotion_df['filename']).sum()
        match_percentage = (matches / total_records) * 100 if total_records > 0 else 0
        emotion_matching[emotion] = match_percentage

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple']

    # Create a larger figure (width, height) in inches
    plt.figure(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.18)  # Adjust bottom margin

    bars = plt.bar(emotion_matching.keys(), emotion_matching.values(), color=colors)

    plt.xlabel('Emotions')
    plt.ylabel('Percentage of matching emotions')
    plt.title(f'Matching emotions for dataframe: {identifier}')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility

    # Adding the percentage values on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, round(yval, 2), ha='center', va='bottom')

    plt.show()
    
