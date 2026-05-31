import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

font_path = "/Users/pavanitonape/Library/Fonts/NotoColorEmoji.ttf"
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Noto Color Emoji'

fig, ax = plt.subplots()
ax.text(0.5, 0.5, "😀 😃 😄 😁 😂 🤩 🚀", fontsize=48, ha='center', va='center')
plt.show()
