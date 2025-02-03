import json
import pandas as pd

data_list = []
grouped = claims_with_flag.groupby('member_id', group_keys=False)

for individual_id, df_individual in grouped:
    # Sort each individual's records by date
    df_individual = df_individual.sort_values('dt').reset_index(drop=True)
    icd_summary = {}
    
    # Iterate over each row so that each date (dt) produces an input text
    for idx, row in df_individual.iterrows():
        # Convert the current row's date to a datetime object and formatted string
        current_dt = pd.to_datetime(row['dt'])
        current_date_str = current_dt.strftime("%Y-%m-%d")
        
        icd = row['icd_cd']
        
        # Update the ICD summary dictionary for the current ICD code
        if icd not in icd_summary:
            icd_summary[icd] = {
                'first_date': current_dt,
                'recent_date': current_dt,
                'total_count': 1
            }
        else:
            icd_summary[icd]['recent_date'] = current_dt
            icd_summary[icd]['total_count'] += 1
        
        # Sort the ICD codes by their most recent date in descending order
        sorted_icd = sorted(icd_summary.items(),
                            key=lambda item: item[1]['recent_date'],
                            reverse=True)
        
        # Build the diagnosis history using the current date as the index date.
        # The time differences (in months) are computed relative to the current_dt.
        diagnosis_history = []
        for code, stats in sorted_icd:
            months_since_first = round((current_dt - stats['first_date']).days / 30)
            months_since_recent = round((current_dt - stats['recent_date']).days / 30)
            diagnosis_history.append(f"{code}|{months_since_first}|{months_since_recent}|{stats['total_count']}")
        
        # Create the input text dictionary
        input_text_dict = {
            'diagnosis_history': diagnosis_history,
            'index_date': current_date_str
        }
        input_text = json.dumps(input_text_dict, separators=(',', ':'))
        label = str(row['inpatient_flag'])
        
        # Append the result for the current row (i.e., each individual at each dt)
        data_list.append({
            'input_text': input_text,
            'label': label
        })

dataset = pd.DataFrame(data_list)
print(dataset.head(5))