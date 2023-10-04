# practise
python
Copy code
features = ['feature1', 'feature2', ...]  # list all your 4000 features
queries = []

for i in range(len(features)):
    for j in range(i+1, len(features)):
        query = f"SELECT CORR({features[i]}, {features[j]}) as correlation_{features[i]}_{features[j]} FROM `your_project_id.your_dataset_id.your_table_id`"
        queries.append(query)

# You'll now have a list of queries in `queries`
