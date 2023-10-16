import yaml
import io
import openml

# Read YAML file
with open("ag.yaml", 'r') as stream:
    data_loaded = yaml.safe_load(stream)

new_d = []
for d in data_loaded:
    d['openml_task_id']
    task = openml.tasks.get_task(d['openml_task_id'], download_splits=False, download_data=False, download_qualities=False, download_features_meta_data=False)
    if (task.task_type != 'Supervised Classification'): #  and (len(task.class_labels) == 2)
        new_d.append(d)

# Write YAML file
print(len(new_d))
with io.open('ag_reg.yaml', 'w', encoding='utf8') as outfile:
    yaml.dump(new_d, outfile, default_flow_style=False, allow_unicode=True)
