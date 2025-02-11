import database_utils
import image_utils

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

print("Loading database...")
database = database_utils.load_database("database")
print("Database loaded.")

print("\nCensoring images...")
image_utils.censor_directory("data/censor_test", "data/censor_output", database)
print("Images censored.")