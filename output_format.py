# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 11:30:24 2025

@author: THINKPAD
"""
import pandas as pd

label_file_path = r"E:/ufl/xue/videos/output/CGS_R2B3_N2S_West_45fpm_60fps_4634/3_parts_tracking/strawberry_parts_counts.xlsx"
df = pd.read_excel(label_file_path)

df['sort_key'] = df['Video'].str.extract(r'_(\d+)\.mp4$')[0].astype(int)
df.sort_values('sort_key', inplace=True)
df.drop(columns=['sort_key'], inplace=True)

output_file_path = r"E:/ufl/xue/videos/output/CGS_R2B3_N2S_West_45fpm_60fps_4634/3_parts_tracking/strawberry_parts_counts.xlsx"
df.to_excel(output_file_path, index=False)