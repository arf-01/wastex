import re

with open('classifier/models.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove DatasetVersion, VersionEntry, DatasetClass classes
content = re.sub(r'# -- Dataset versioning --.*?# -- Uploaded images --', '# -- Uploaded images --', content, flags=re.DOTALL)

# Remove TrainingRun class
content = re.sub(r'# -- Training runs --.*?# -- Application Settings \(Installation Configuration\) --', '# -- Application Settings (Installation Configuration) --', content, flags=re.DOTALL)

# Remove Image model's Dataset linkage section
content = re.sub(r'\s+# -- Dataset linkage --.*?# -- Cloud Sync --', '\n    # -- Cloud Sync --', content, flags=re.DOTALL)

# Remove the staging area query index from Image Meta
content = re.sub(r'\s+# Composite index for the staging area query.*?models\.Index\(.*?name=\'idx_image_staging\',.*?\),', '', content, flags=re.DOTALL)

with open('classifier/models.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('models.py cleaned')
