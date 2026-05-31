from streamlit import columns
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji



def fetch_stats(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # fetch the number of media messages
    num_media_messages = df[df['message'] =='Media files shared'].shape[0]


    #fetch the number of links shared

    num_links = df['message'].str.contains('http').sum()


    return num_messages,len(words),num_media_messages,num_links

def most_busy_users(df):
    x = df['user'].value_counts().head()
    df=round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index':'name','user':'percent'})
    return x,df

def create_wordcloud(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    wc = WordCloud(width = 800, height = 800,min_font_size=10,background_color='white')
    df_wc=wc.generate(df['message'].str.cat(sep=' '))
    return df_wc

def most_common_words(selected_user,df):
    f = open('/Users/pavanitonape/Downloads/stop_hinglish.txt', 'r', encoding='utf-8')
    stop_words=f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

# to remove group notification and media omitted

    temp=df[df['user'] != 'group_notification']
    temp=temp[temp['message']!='<media omitted>\n']

    words=[]
    for message in temp['message']:
       for word in message.lower().split():
           if word not in stop_words:
               words.append(word)

    most_common_df=pd.DataFrame(Counter(words).most_common(10))
    return  most_common_df

def emoji_helper(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp=df[df['user'] != 'group_notification']
    temp=temp[temp['message']!='<media omitted>\n']

    emojis = []
    for message in temp['message']:
        emoji_list = emoji.emoji_list(message)  # ✅ BETTER DETECTION
        for em in emoji_list:
            emojis.append(em['emoji'])

    emoji_df=pd.DataFrame(Counter(emojis).most_common(20),columns=['emoji','count'])
    return emoji_df


