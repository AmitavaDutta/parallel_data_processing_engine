# 🚀 Project Pulse & Collaborator Stats

> This dashboard updates automatically using GitHub's real-time API. No scripts or manual updates required.

---

## 👥 The Development Team

<table border="0">
  <tr>
    <td align="center" width="50%">
      <h3>🔥 Amitava Dutta</h3>
      <img src="https://github-readme-stats.vercel.app/api?username=AmitavaDutta&show_icons=true&theme=tokyonight&hide_border=true&count_private=true" width="400"/><br/>
      <img src="https://github-readme-activity-graph.vercel.app/graph?username=AmitavaDutta&theme=tokyonight&area=true&hide_border=true" width="400"/>
    </td>
    <td align="center" width="50%">
      <h3>⚡ S Yashvita</h3>
      <img src="https://github-readme-stats.vercel.app/api?username=SYashvita&show_icons=true&theme=tokyonight&hide_border=true&count_private=true" width="400"/><br/>
      <img src="https://github-readme-activity-graph.vercel.app/graph?username=SYashvita&theme=tokyonight&area=true&hide_border=true" width="400"/>
    </td>
  </tr>

  <tr>
    <td align="center" width="50%">
      <h3>🌟 Sipra Subhadarsini Sahoo</h3>
      <img src="https://github-readme-stats.vercel.app/api?username=Sipra-S&show_icons=true&theme=tokyonight&hide_border=true&count_private=true" width="400"/><br/>
      <img src="https://github-readme-activity-graph.vercel.app/graph?username=Sipra-S&theme=tokyonight&area=true&hide_border=true" width="400"/>
    </td>
    <td align="center" width="50%">
      <h3>💎 Bhavini</h3>
      <img src="https://github-readme-stats.vercel.app/api?username=bhaviniraina&show_icons=true&theme=tokyonight&hide_border=true&count_private=true" width="400"/><br/>
      <img src="https://github-readme-activity-graph.vercel.app/graph?username=bhaviniraina&theme=tokyonight&area=true&hide_border=true" width="400"/>
    </td>
  </tr>
</table>

---

## 📊 Repository Insights

To keep the momentum going, here is the language breakdown for our **Parallel Data Processing Engine**:

<p align="center">
  <img src="https://github-readme-stats.vercel.app/api/pin/?username=AmitavaDutta&repo=parallel_data_processing_engine&theme=tokyonight&show_owner=true" />
  <img src="https://github-readme-stats.vercel.app/api/top-langs/?username=AmitavaDutta&layout=compact&theme=tokyonight&hide_border=true" />
</p>

---
**Tip:** These graphs reflect *all* GitHub activity. If you want to see specific trends for just this repository, ensure everyone is committing to the `main` or `develop` branches regularly!name: Update Project Statistics
on:
  push:
    branches: [ main ]
  schedule:
    - cron: '0 0 * * 0' # Runs every Sunday at midnight
  workflow_dispatch:

jobs:
  stats:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0 # Necessary for analyzing full commit history

      - name: Generate Statistics
        run: |
          echo "# Project Statistics" > project_statistics.md
          echo "Last updated: $(date)" >> project_statistics.md
          echo "" >> project_statistics.md
          
          # Calculate start date and weeks
          start_date=$(git log --reverse --format=%at | head -1)
          echo "## Commits per Week (by Contributor)" >> project_statistics.md
          echo "| Contributor | Week Number | Commits |" >> project_statistics.md
          echo "| :--- | :--- | :--- |" >> project_statistics.md
          
          # Generate data
          git log --format='%aN|%at' | while IFS='|' read -r author ts; do
            week=$(( (ts - start_date) / 604800 + 1 ))
            echo "$author|Week $week"
          done | sort | uniq -c | awk '{print "| " $3 " | " $4 " | " $1 " |"}' >> project_statistics.md

      - name: Commit and Push
        run: |
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action"
          git add project_statistics.md
          git commit -m "docs: update project statistics [skip ci]" || exit 0
          git push
