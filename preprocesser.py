import re
import pandas as pd


def preprocess(data):
    # FIXED: Better regex that preserves emojis
    pattern = r'\[(\d{2}/\d{2}/\d{2}, \d{1,2}:\d{2}:\d{2}(?:\u202F|\s)?[AP]M)\] ([\w\W]+?): ([\w\W]+)'

    df = pd.DataFrame(columns=['date', 'user', 'message'])

    for match in re.finditer(pattern, data):
        date_str = match.group(1)
        user = match.group(2)
        message = match.group(3)

        try:
            date = pd.to_datetime(date_str, format='[%d/%m/%y, %I:%M:%S\u202f%p]')
            df = pd.concat([df, pd.DataFrame({
                'date': [date],
                'user': [user],
                'message': [message]
            })], ignore_index=True)
        except:
            # Fallback for group notifications
            df = pd.concat([df, pd.DataFrame({
                'date': [pd.NaT],
                'user': ['group_notification'],
                'message': [match.group(0)]
            })], ignore_index=True)

    # Add time columns
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['second'] = df['date'].dt.second

    return df
