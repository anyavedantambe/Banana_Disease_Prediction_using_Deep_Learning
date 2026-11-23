import pandas as pd

new = pd.read_csv("splits/splits_blocks100.csv")
new_test = set(new[new.split == "test"].rel_path)
old_test = set(pd.read_csv("nn_pairs.csv").test_file)
print(new_test == old_test, len(new_test), len(old_test))