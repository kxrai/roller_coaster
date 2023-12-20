import pandas as pd
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import sys
import re
from sympy import simplify

def replace_pi_with_sp_pi(expression):
    return re.sub(r'\\bpi\\b', 'sp.pi', expression)

def sympify_with_pi_handling(formula):
    formula = replace_pi_with_sp_pi(formula)
    return sp.sympify(formula, locals={'sp.pi': sp.pi})

def evaluate_expression(expression):
    expression = replace_pi_with_sp_pi(expression)
    return sp.sympify(expression).evalf()

def validate_csv(df):
    tolerance = 1e-6  # Define a small tolerance for smoothness checking

    for i, row in df.iterrows():
        formula = replace_pi_with_sp_pi(row['formula'])
        start_x = evaluate_expression(row['start_x'])
        end_x = evaluate_expression(row['end_x'])

        if end_x <= start_x:
            raise ValueError(f"End value of x must be greater than start value in row {i}")

        if i > 0:
            prev_end_x = evaluate_expression(df.iloc[i-1]['end_x'])
            if start_x != prev_end_x:
                raise ValueError(f"Start value of x in row {i} does not match end value of x in previous row")

            prev_formula = replace_pi_with_sp_pi(df.iloc[i-1]['formula'])
            current_formula_sympy = sympify_with_pi_handling(formula)
            prev_formula_sympy = sympify_with_pi_handling(prev_formula)

            # Calculate and simplify derivatives
            prev_derivative = simplify(sp.diff(prev_formula_sympy, sp.Symbol('x')).subs(sp.Symbol('x'), prev_end_x))
            current_derivative = simplify(sp.diff(current_formula_sympy, sp.Symbol('x')).subs(sp.Symbol('x'), start_x))

            # Check if derivatives differ by more than the tolerance
            if abs(prev_derivative - current_derivative) > tolerance:
                raise ValueError(f"Function not smooth between row {i-1} and row {i}")

    print("CSV validation passed.")
     
def plot_segment(formula, start_x, end_x, ax):
    # Convert the formula to a sympy expression
    expr = sympify_with_pi_handling(formula)

    # Evaluate start_x and end_x to numerical values and convert to float
    start_x_num = float(evaluate_expression(start_x))
    end_x_num = float(evaluate_expression(end_x))

    # Create a lambda function for the expression
    f = sp.lambdify(sp.Symbol('x'), expr, 'numpy')

    # Generate x values
    x_vals = np.linspace(start_x_num, end_x_num, 1000)

    # Compute y values
    y_vals = f(x_vals)

    # Plot the segment
    ax.plot(x_vals, y_vals, label=formula)

    print("plot_segment completed")


def main():
    file_name = 'lewin_input.csv'
    df = pd.read_csv(file_name) 
    validate_csv(df)

    fig, ax = plt.subplots()

    for index, row in df.iterrows():
        plot_segment(row['formula'], row['start_x'], row['end_x'], ax)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    output_file = 'roller_coaster.svg'
    plt.savefig(output_file, format='svg')
    plt.show()
    print(f"Plot saved to {output_file}")

if __name__ == '__main__':
    main()
