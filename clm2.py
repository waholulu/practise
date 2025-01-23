def merge_intervals(intervals):
    # Sort by the start date
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = []
    for interval in intervals:
        if not merged or interval[0] > merged[-1][1]:
            merged.append(interval)
        else:
            # Overlapping intervals, so we update the end if needed
            merged[-1] = (merged[-1][0], max(merged[-1][1], interval[1]))
    return merged

def mark_inpatient_flag(df, six_months=pd.Timedelta(days=180)):
    df = df.copy()
    df["inpatient_flag"] = 0

    inpatient_code = "INPAT"
    df_list = []

    for ind_id, sub_df in df.groupby("individual_id", group_keys=False):
        sub_df = sub_df.sort_values("dt").copy()

        inpat_dates = sub_df.loc[sub_df['icd_cd'] == inpatient_code, 'dt'].sort_values()
        if len(inpat_dates) > 0:
            # Create intervals: [inpatient_date - 6 months, inpatient_date]
            intervals = [(d - six_months, d) for d in inpat_dates]

            # Optionally merge overlapping intervals
            merged_intervals = merge_intervals(intervals)

            claim_dates = sub_df['dt'].values
            inpatient_flag = np.zeros(len(sub_df), dtype=int)

            # Single-pass approach to mark intervals
            i = 0  # pointer to intervals
            for j, claim_dt in enumerate(claim_dates):
                # Advance the interval pointer if we've moved past the current interval
                while i < len(merged_intervals) and claim_dt > merged_intervals[i][1]:
                    i += 1
                # Check if we're within the current interval
                if i < len(merged_intervals):
                    start_int, end_int = merged_intervals[i]
                    if start_int <= claim_dt <= end_int:
                        inpatient_flag[j] = 1

            sub_df["inpatient_flag"] = inpatient_flag

        df_list.append(sub_df)

    return pd.concat(df_list, ignore_index=True)


# --- Mark the inpatient_flag in a separate step ---
claims_with_flag = mark_inpatient_flag(claims_df, six_months=pd.Timedelta(days=180))

# Now claims_with_flag has an "inpatient_flag" column
print(claims_with_flag.head(20))


#############


import pandas as pd
import json

data_list = []

# Group by individual_id
grouped = claims_with_flag.groupby('individual_id', group_keys=False)

for individual_id, df_individual in grouped:
    # Sort by dt
    df_individual = df_individual.sort_values('dt').reset_index(drop=True)

    # Running summary for each ICD code
    # { icd_cd: {"first_date": ..., "recent_date": ..., "total_count": ...} }
    icd_summary = {}

    for idx, row in df_individual.iterrows():
        # Current row details
        icd = row['icd_cd']
        current_date_str = row['dt'].strftime('%Y-%m-%d')  # this will be our 'index_date'
        
        # Update summary for this ICD
        if icd not in icd_summary:
            icd_summary[icd] = {
                'first_date': current_date_str,
                'recent_date': current_date_str,
                'total_count': 1
            }
        else:
            icd_summary[icd]['recent_date'] = current_date_str
            icd_summary[icd]['total_count'] += 1

        # Build the list-of-dicts from our running summary
        diagnosis_history = []
        for code, stats in icd_summary.items():
            diagnosis_history.append({
                'icd_cd': code,
                'first_date': stats['first_date'],
                'recent_date': stats['recent_date'],
                'total_count': stats['total_count']
            })
        
        # OPTIONAL: sort diagnosis_history by 'icd_cd' or leave as is
        # diagnosis_history.sort(key=lambda x: x['icd_cd'])

        # Create the final dictionary for JSON
        # Note the new key 'index_date' at the top level
        input_text_dict = {
            'diagnosis_history': diagnosis_history,
            'index_date': current_date_str
        }

        # Convert dictionary to JSON
        input_text = json.dumps(input_text_dict, separators=(',', ':'))
        
        # Label
        label = str(row['inpatient_flag'])
        
        # Append to the data list
        data_list.append({
            'input_text': input_text,
            'label': label
        })

# Convert to a DataFrame
dataset = pd.DataFrame(data_list)
print(dataset.head(5))


# Find the length of each input_text
dataset['input_text_length'] = dataset['input_text'].apply(len)

# Find the row with the longest input text
longest_row = dataset.loc[dataset['input_text_length'].idxmax()]

# Print the length and the input text
print(f"Length of the longest input_text: {longest_row['input_text_length']}")
print(f"Longest input_text: {longest_row['input_text']}")
