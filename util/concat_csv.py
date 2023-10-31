import os
import pandas as pd

# Define file paths for the three CSV files
body_file = "/home/aiotlab3/RISE/Lab-MA/DucAnh/SOTitle/data/Python/test_body.csv"
code_file = "/home/aiotlab3/RISE/Lab-MA/DucAnh/SOTitle/data/Python/test_code.csv"
title_file = "/home/aiotlab3/RISE/Lab-MA/DucAnh/SOTitle/data/Python/test_title.csv"

df_body = pd.read_csv(body_file)
df_code = pd.read_csv(code_file)
df_title = pd.read_csv(title_file)

merged_df = pd.concat([df_body, df_code, df_title], axis=1, keys=['body', 'code', 'title'])

output_file = "/home/aiotlab3/RISE/Lab-MA/DucAnh/transformer_copy/data/Python/test.csv"

# Check if the output file already exists
if not os.path.exists(output_file):
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Save the merged DataFrame to the output file
merged_df.to_csv(output_file, index=False)
