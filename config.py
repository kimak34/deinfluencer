import database_utils

names = []
name = input("Please enter names of people you would like to block one at a time (q to quit): ")
while name != "q" and name != "Q":
    names.append(name)
    name = input("Please enter names of people you would like to block one at a time (q to quit): ")

print("\nSaving list of names...")
database_utils.save_database("names", names)
print("Names saved.")

print("\nPopulating database...")
database = database_utils.populate_database(names, 5, "images")
print("Database populated.")

print("\nSaving database")
database_utils.save_database("database", database)
print("Database saved.")