import json
import pandas as pd

data_list = []
grouped = claims_with_flag.groupby('member_id', group_keys=False)

for individual_id, df_individual in grouped:
    # Sort the individual's records by date
    df_individual = df_individual.sort_values('dt').reset_index(drop=True)
    icd_summary = {}

    # Process each row to collect first and most recent dates per ICD code
    for idx, row in df_individual.iterrows():
        icd = row['icd_cd']
        row_date = pd.to_datetime(row['dt'])
        # Keep the last row's date as the overall index_date for this individual.
        index_date = row_date  
        index_date_str = row_date.strftime("%Y-%m-%d")

        if icd not in icd_summary:
            icd_summary[icd] = {
                'first_date': row_date,
                'recent_date': row_date,
                'total_count': 1
            }
        else:
            # Because the data is sorted, any new appearance of the ICD code is more recent
            icd_summary[icd]['recent_date'] = row_date
            icd_summary[icd]['total_count'] += 1

    # Sort the ICD codes based on the recent_date in descending order:
    sorted_icd = sorted(icd_summary.items(),
                        key=lambda item: item[1]['recent_date'],
                        reverse=True)

    diagnosis_history = []
    # Build the diagnosis history in the sorted order.
    for code, stats in sorted_icd:
        months_since_first = round((index_date - stats['first_date']).days / 30)
        months_since_recent = round((index_date - stats['recent_date']).days / 30)
        diagnosis_history.append(f"{code}|{months_since_first}|{months_since_recent}|{stats['total_count']}")

    input_text_dict = {
        'diagnosis_history': diagnosis_history,
        'index_date': index_date_str
    }

    input_text = json.dumps(input_text_dict, separators=(',', ':'))
    # Using the inpatient_flag from the last processed row for this individual
    label = str(row['inpatient_flag'])

    data_list.append({
        'input_text': input_text,
        'label': label
    })

dataset = pd.DataFrame(data_list)
print(dataset.head(5))