# project using final fantasy 7 text dump csv
# aggregates lines of text according to playable characters
# further filters by expressions used by characters indicating happy, sadness
# characters ranked by emotion value using dataplots
import pandas as panda
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
panda.set_option('display.max_rows', None)
panda.set_option('display.max_columns', None)
panda.set_option('display.width', None)
panda.set_option('display.max_colwidth', None)
ff7 = panda.read_csv('C:/Users/Shoeb Mohammed/Documents/ff7-script.csv')
print(ff7.head()) #test connectivity

whoSpeaks = ff7['Character']
print(set((whoSpeaks)))

heroMap = { #mapping different labels to get more accurate results of character quotes
    'flower girl': 'Aeris',
    'aerith': 'Aeris',
    'aeris': 'Aeris',
    'cloud': 'Cloud',
    'cloudaftermeteor': 'Cloud',
    'tifa': 'Tifa',
    'barret': 'Barret',
    'yuffie': 'Yuffie',
    'cid': 'Cid',
    'redxiiibeforeinthelandofthestudyofplanetlife': 'RedXIII',
    'redxiii': 'RedXIII',
    'red xiii': 'RedXIII',
    'vincent': 'Vincent',
    'cait sith': 'Cait Sith',
    'caitsith': 'Cait Sith'
}
ff7['NormalizedCharacter'] = ff7['Character'].str.lower().str.strip().map(heroMap) #strip hero names
filtered = ff7[ff7['NormalizedCharacter'].notna()]

speakingChars = (
    filtered.groupby('NormalizedCharacter')
        .size()
        .reset_index(name='LineCount')
        .sort_values(by='LineCount', ascending=False
    ))

speakingChars = speakingChars.rename(columns={'NormalizedCharacter': 'Hero'})
speakingChars2 = speakingChars.copy()
print(speakingChars)
print("")

#small set of words to test character lines against
negative_words = [
    'cry', 'tears', 'sad', 'weep', 'grief', 'mourn', 'sorrow',
    'hate', 'angry', 'furious', 'rage', 'yell', 'mad', 'annoyed',
    'why', 'hopeless', "can’t", 'alone', 'dark', 'pain', 'suffer'
]

positive_words = [
    'laugh', 'giggle', 'chuckle', 'love', 'adore', 'smile',
    'grin', 'beam', 'happy', 'cheer', 'kiss', 'eager',
    'joy', 'pleased'
]
#make a pattern distinguishing lines which contain +/- words
patterna = pattern_positive = r'\b(?:' + '|'.join(positive_words) + r')\b'
patternb = pattern_negative = r'\b(?:' + '|'.join(negative_words) + r')\b'
ff7['DialogueLower'] = ff7['Dialogue'].str.lower()
mask_a = ff7['DialogueLower'].str.contains(patterna, na=False)
mask_b = ff7['DialogueLower'].str.contains(patternb, na=False)
happyLines = ff7[mask_a]  # lines containing happy words
sadLines = ff7[mask_b] #lines containing sad words


happyCounts = happyLines.groupby('Character').size().reset_index(name='KeyWordLineCount').sort_values(
    by='KeyWordLineCount', ascending=False)

sadCounts = sadLines.groupby('Character').size().reset_index(name='KeyWordLineCount').sort_values(
    by='KeyWordLineCount', ascending=False)


happyCounts['Hero'] = happyCounts['Character'].str.lower().str.strip().map(heroMap)
sadCounts['Hero'] = sadCounts['Character'].str.lower().str.strip().map(heroMap)

hc2 = happyCounts.copy()
sc2 = sadCounts.copy()

# After mapping to Hero, drop NaNs and group again
happyCounts = (
    happyCounts.dropna(subset=['Hero'])
    .groupby('Hero', as_index=False)['KeyWordLineCount']
    .sum()
)

sadCounts = (
    sadCounts.dropna(subset=['Hero'])
    .groupby('Hero', as_index=False)['KeyWordLineCount']
    .sum()
)

hc2 = (
    hc2.dropna(subset=['Hero'])
    .groupby('Hero')
    .size()
    .reset_index(name='HappyLineCount')
)

sc2 = (
    sc2.dropna(subset=['Hero'])
    .groupby('Hero')
    .size()
    .reset_index(name='SadLineCount')
)


ff7HappyMerged = speakingChars.merge(happyCounts[['Hero', 'KeyWordLineCount']], on='Hero', how='left')
ff7HappyMerged['KeyWordLineCount'] = ff7HappyMerged['KeyWordLineCount'].fillna(0).astype(int)
ff7HappyMerged['EmotionLinePercent'] = (
        (ff7HappyMerged['KeyWordLineCount'] / ff7HappyMerged['LineCount']) * 100
).round(2)
ff7HappyMerged = ff7HappyMerged.rename(columns={'KeyWordLineCount': 'HappyWordLineCount'})
ff7HappyMerged = ff7HappyMerged.rename(columns={'EmotionLinePercent': 'HappyPercent'})


ff7SadMerged = speakingChars2.merge(sadCounts[['Hero', 'KeyWordLineCount']], on='Hero', how='left')
ff7SadMerged['KeyWordLineCount'] = ff7SadMerged['KeyWordLineCount'].fillna(0).astype(int)
ff7SadMerged['EmotionLinePercent'] = (
        (ff7SadMerged['KeyWordLineCount'] / ff7SadMerged['LineCount']) * 100
).round(2)
ff7SadMerged = ff7SadMerged.rename(columns={'KeyWordLineCount': 'SadWordLineCount'})
ff7SadMerged = ff7SadMerged.rename(columns={'EmotionLinePercent': 'SadPercent'})

def assign_mood_happy(percent):
    if percent == 0:
        return '💔'
    elif percent <= 2:
        return '😩'
    elif percent <= 5:
        return '🙂'
    elif percent <= 8:
        return '🙂'
    else:
        return '😎'

def assign_mood_sad(percent):
    if percent == 0:
        return '😎'
    elif percent <= 2:
        return '🙂'
    elif percent <= 5:
        return '😢'
    elif percent <= 8:
        return '😩'
    else:
        return '💔'

ff7HappyMerged['EmotionMeter'] = ff7HappyMerged['HappyPercent'].apply(assign_mood_happy)
ff7SadMerged['EmotionMeter'] = ff7SadMerged['SadPercent'].apply(assign_mood_sad)

print("")
print(ff7HappyMerged)
print("")
print(ff7SadMerged)

#merge
# Merge on Hero
emotion_df = ff7HappyMerged.merge(
    ff7SadMerged[['Hero', 'SadWordLineCount', 'SadPercent']],
    on='Hero',
    how='outer'
)

# Fill NaNs with 0
emotion_df[['HappyWordLineCount', 'HappyPercent', 'SadWordLineCount', 'SadPercent']] = emotion_df[
    ['HappyWordLineCount', 'HappyPercent', 'SadWordLineCount', 'SadPercent']
].fillna(0)

print("")
emotion_df = emotion_df.rename(columns={'EmotionMeter': 'HappyEmotionMeter'})
emotion_df['SadEmotionMeter'] = ff7SadMerged['EmotionMeter']

print(emotion_df)

#######################################################################################################################################
#Visualization Code

#######################################################################################################################################
emotion_palette = {
    'HappyPercent': '#00ffcc',  # Mako glow
    'SadPercent': '#b30000'     # Shinra red
}
character_colors = {
    'Cloud': '#00ffcc',
    'Tifa': '#ffb6c1',
    'Barret': '#b30000',
    'Aeris': '#dda0dd',
    'Vincent': '#2e2e2e',
    'Yuffie': '#66ff66',
    'RedXIII': '#ff6600',
    'Cid': '#cccccc',
    'Cait Sith': '#9999ff'
}

plt.figure(figsize=(10, 6))
sns.barplot(data=emotion_df, x='Hero', y='HappyPercent', palette=character_colors)
plt.title('😊 Happy Line Percentage by Hero')
plt.ylabel('Percent of Lines with Happy Words')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#bar chart - sad
plt.figure(figsize=(10, 6))
sns.barplot(data=emotion_df, x='Hero', y='SadPercent', palette=character_colors)
plt.title('😊 Sad Line Percentage by Hero')
plt.ylabel('Percent of Lines with Sad Words')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#bar chart - stacked
# Melt the DataFrame for Seaborn
emotion_melted = emotion_df.melt(
    id_vars='Hero',
    value_vars=['HappyPercent', 'SadPercent'],
    var_name='EmotionType',
    value_name='Percent'
)

plt.figure(figsize=(10, 6))
sns.barplot(data=emotion_melted, x='Hero', y='Percent', hue= 'EmotionType', palette=emotion_palette)
plt.title('Emotional Expression by Hero')
plt.ylabel('Percent of Lines')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



#stackedBar
# Plot
emotion_df = emotion_df.sort_values(by='Hero')
heroes = emotion_df['Hero']
happy = emotion_df['HappyPercent']
sad = emotion_df['SadPercent']
# Set up bar positions
x = np.arange(len(heroes))
width = 0.6

# Colors
happy_color = '#00ffcc'  # Mako glow
sad_color = '#b30000'    # Shinra red
plt.figure(figsize=(12, 6))
plt.bar(x, happy, width, label='Happy 😊', color=happy_color)
plt.bar(x, sad, width, bottom=happy, label='Sad 😢', color=sad_color)

# Labels and styling
plt.xticks(x, heroes, rotation=45)
plt.ylabel('Emotion Line %')
plt.title('Stacked Emotional Expression by Hero')
plt.legend()
plt.tight_layout()
plt.show()


#heatMap
heatmap_data = emotion_df.set_index('Hero')[['HappyPercent', 'SadPercent']]

plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Emotion Intensity Heatmap')
plt.tight_layout()
plt.show()

#scatterPlot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=emotion_df, x='HappyPercent', y='SadPercent', hue='Hero', s=100, palette=character_colors)
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.title('Emotional Polarity: Happy vs Sad')
plt.xlabel('Happy Line %')
plt.ylabel('Sad Line %')
plt.tight_layout()
plt.show()

#pie
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
plt.pie(hc2['HappyLineCount'], labels=hc2['Hero'], autopct='%1.1f%%', startangle=140)
plt.title('😊 Distribution of Happy Lines by Hero')
plt.axis('equal')  # Equal aspect ratio ensures the pie is circular
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 8))
plt.pie(sc2['SadLineCount'], labels=sc2['Hero'], autopct='%1.1f%%', startangle=140)
plt.title('😢 Distribution of Sad Lines by Hero')
plt.axis('equal')
plt.tight_layout()
