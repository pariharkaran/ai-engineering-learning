name = "Karan"

print(f"Hello, {name}!")

my_details = {
    "name": "Karan",
    "age": 25,
    "role": "Software Engineer"
};

print(f"My name is {my_details['name']}, I am {my_details['age']} years old and I work as a {my_details['role']}.")   

fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    if(fruit == "apple"):
        print(f"{fruit.capitalize()}s are my favorite fruit!")
    else:
        print(f"{fruit.capitalize()}s are also great!")

def greet_user(name):
    return f"Hello, {name}! Welcome to the program."

greeting = greet_user("Karan")
print(greeting)