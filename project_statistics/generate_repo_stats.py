import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import os
from datetime import datetime

# 1. Get Repo-Specific Git Logs
cmd = 'git log --format="%ad,%an" --date=iso'
log_data = subprocess.check_output(cmd, shell=True).decode('utf-8')

# 2. Process Data
df = pd.DataFrame([line.split(',', 1) for line in log_data.replace(' ', ',', 1).split('\n') if line], 
                  columns=['date', 'author'])
df['date'] = pd.to_datetime(df['date'], utc=True)

# 3. Calculate "Week X" from Project Start
start_date = df['date'].min()
df['week_num'] = df['date'].apply(lambda x: (x - start_date).days // 7 + 1)
df['week_label'] = df['week_num'].apply(lambda x: f"Week {x}")

# 4. Group by Week and Author
stats = df.groupby(['week_num', 'week_label', 'author']).size().unstack(fill_value=0)
stats = stats.reset_index().set_index('week_label').drop(columns='week_num')

# 5. Visualization
plt.figure(figsize=(10, 5))
plt.style.use('dark_background') 
for column in stats.columns:
    plt.plot(stats.index, stats[column], marker='o', label=column, linewidth=2)

plt.title('Parallel Data Processing Engine: Contribution Timeline', fontsize=14, pad=20)
plt.ylabel('Commits')
plt.xlabel('Project Timeline')
plt.legend(frameon=False)
plt.grid(axis='y', alpha=0.1)
plt.tight_layout()

# 6. Save path relative to project root (since Action runs from root)
save_path = os.path.join('project_statistics', 'repo_stats.svg')
plt.savefig(save_path, transparent=True)
