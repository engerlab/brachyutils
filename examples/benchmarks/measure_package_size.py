import os
import pkg_resources
import pandas as pd

def get_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
    return total_size

data = []
for dist in pkg_resources.working_set:
    try:
        package_path = os.path.join(dist.location, dist.project_name)
        size_bytes = get_size(package_path)
        size_mb = size_bytes / 1024 # Convert to MB
        data.append([dist.project_name, size_mb])
    except Exception:
        data.append([dist.project_name, None])

df = pd.DataFrame(data, columns=['Package Name', 'Size (MB)'])
df.to_csv('python_package_sizes.csv', index=False)

print("Package sizes saved to python_package_sizes.csv")
