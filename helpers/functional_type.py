from helpers.data_loader import DataLoader
import re
import pandas as pd


FUNCTIONAL_TYPES = {
    'method_declaration': r"^(?:@[a-zA-Z]*|public|protected|private|abstract|static|final|strictfp|synchronized|native|\s)*\s*[\w\<\>\[\], ]*\(",
    'class_declaration': r"^(?:@[a-zA-Z]*|public|protected|private|abstract|static|final|strictfp|synchronized|native|\s)*(class|interface|@interface)\s",
    'assignment': r"^[\s]*\w[\s\w[\]\*<,\?>\.]*=",
    'method_call': r"^[\s]*[\w.]+\(",
    'return': r"^[\s]*return(\s|;)",
    'loop': r"^[\s]*(for|while)\s",
    'conditional': r"^\s*(if|switch|else)(\s|\()",
    'catch': r"^[\s]*}?[\s]*catch",
    'var_declaration': r"^(?:@[a-zA-Z]*|public|protected|private|abstract|static|final|strictfp|synchronized|native|\s)*\w+(<[a-zA-Z<>.\?]*>)?\s+\w+\s*;",
    'package_import': r"\s*(package|import)\s",
    'loop_exit': "\s*(continue|break)(\s|;)"
}
# requires_re = r"\s*requires\s"
# requires_name = 'requires'
#requires not handled

data = DataLoader.load_data()
functional_types = pd.DataFrame(columns=["functional_type"], data=[])

for index, row in data.iterrows():
    curr_code = row['code']
    val = 'empty'
    for key in FUNCTIONAL_TYPES:
        reg_exp = re.compile(FUNCTIONAL_TYPES[key])
        if reg_exp.match(curr_code):
            val = key
            break
    functional_types.loc[index] = val

functional_types.to_csv('./../data/functional_types.csv')
print(data)