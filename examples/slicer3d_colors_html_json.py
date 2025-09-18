from bs4 import BeautifulSoup
import json
from pathlib import Path
import requests
# pth_html = Path("admin/constants/Slicer3_2010 GenericAnatomyColors - Slicer Wiki.html")
pth_html = "https://www.slicer.org/wiki/Slicer3:2010_GenericAnatomyColors"
pth_json = Path("admin/constants/slicer_colors.json")

response = requests.get(pth_html)
html_contents = response.text

soup = BeautifulSoup(html_contents, 'html.parser')
table = soup.find('table')
rows = table.find_all("tr")

color_list = []

# Extract headers
for i, row in enumerate(rows):
    # columns are in the first row
    if i == 0:
        cols = row.find_all("th")
        keys = [col.get_text(strip=True) for col in cols]
        # The keys are: integer_label, text_label, color, notes
    else:
        cols = row.find_all("td")
        vals = [col.get_text(strip=True) for col in cols]
        rgb = vals[2].split("(")[-1].split(")")[0].split(",")[:3] 
        color_list.append(
            {
                keys[0]: int(vals[0]),
                keys[1]: vals[1],
                keys[2]: (int(rgb[0]), int(rgb[1]), int(rgb[2])),
                keys[3]: vals[3],
            }
        )

with open(pth_json, "w") as f:
    json.dump(color_list, f, indent=4)
print(f"Saved {pth_json}")