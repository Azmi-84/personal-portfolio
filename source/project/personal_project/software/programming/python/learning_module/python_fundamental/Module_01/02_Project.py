import marimo

__generated_with = "0.23.14"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from math import pi, sqrt

    return mo, pi


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    1. Calculating the area of a circle
    """)
    return


@app.cell
def _():
    radius = eval(input("Enter the radius of the circle: "))
    return (radius,)


@app.cell
def _(pi, radius):
    if radius > 0:
        area = pi * (radius**2)
        print(
            f"Radius of the circle: {radius} | Area: {area} | Data Type: {isinstance(radius, int)}"
        )
    elif radius == 0:
        print("A circle with radius 0 has no area.")
    else:
        print("Error: Radius cannot be negative or string!")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Simple calculator
    """)
    return


@app.cell
def _():
    first_number = eval(input("Enter the first number: "))
    second_number = eval(input("Enter the second number: "))
    return first_number, second_number


@app.cell
def _():
    print(
        f"Which operation you want? | (1) Addition | (2) Subtraction | (3) Multiplication | (4) Division"
    )
    return


@app.cell
def _():
    choice = eval(input("Give your choice: "))
    return (choice,)


@app.cell
def _(choice, first_number, second_number):
    if choice == 1:
        add = first_number + second_number

        print(
            f"First number is: {first_number} | Second number is: {second_number} | Addition: {first_number + second_number}"
        )

    elif choice == 2:
        sub = first_number - second_number

        print(
            f"First number is: {first_number} | Second number is: {second_number} | Subtraction: {first_number - second_number}"
        )

    elif choice == 3:
        mul = first_number * second_number

        print(
            f"First number is: {first_number} | Second number is: {second_number} | Multiplication: {first_number * second_number}"
        )

    elif choice == 4:
        if second_number > 0:
            div = first_number / second_number

            print(
                f"First number is: {first_number} | Second number is: {second_number} | Division: {first_number / second_number}"
            )

        else:
            print("Denominator can't be zero")
    else:
        print("Wrong input!!!")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Simple prime number checker
    """)
    return


@app.cell
def _():
    val = eval(input("Enter a number: "))
    return (val,)


@app.cell
def _(val):
    flag = 0

    if val <= 1:
        print("Enter number more than 1")

    # for i in range(2, int(sqrt(val)) + 1):
    for i in range(2, int(val**0.5) + 1):
        if val % i == 0:
            flag = 1

    if flag == 0:
        print(f"{val} is a prime number")
    else:
        print(f"{val} is not a prime number")
    return


if __name__ == "__main__":
    app.run()
