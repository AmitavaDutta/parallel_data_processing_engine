name: Update Project Statistics
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
