import pandas as pd
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import sys
import re

def correct_formula(formula):
    # Add multiplication sign where necessary
    corrected_formula = re.sub(r'(\d)([a-zA-Z\(])', r'\1*\2', formula)
    corrected_formula = re.sub(r'(\))(\d)', r'\1*\2', corrected_formula)
    return corrected_formula

def read_and_print_csv(file_name):
    try:
        df = pd.read_csv(file_name)
        print(df)
        return df
    except Exception as e:
        print(f"Error reading the file: {e}")
        sys.exit(1)

def is_continuous(formula1, formula2, at_x, tolerance=1e-6):
    x = sp.Symbol('x')
    # Evaluate the limits from the left and right
    left_limit = sp.limit(formula1, x, at_x, dir='-').evalf()
    right_limit = sp.limit(formula2, x, at_x, dir='+').evalf()
    
    # Check if the difference between left and right limits is within the tolerance
    return abs(left_limit - right_limit) < tolerance

def validate_csv(df):
    x = sp.Symbol('x')
    for i, row in df.iterrows():
        # Correct the formula syntax by adding explicit multiplication
        corrected_formula = correct_formula(row['formula'])
        try:
            formula = sp.sympify(corrected_formula)
        except sp.SympifyError:
            raise ValueError(f"Invalid formula in row {i}: {corrected_formula}")

        start_x = sp.sympify(row['start_x'])
        end_x = sp.sympify(row['end_x'])

        # Check if end_x is greater than start_x
        if end_x <= start_x:
            raise ValueError(f"End value of x must be greater than start value in row {i}")

        # Skip continuity and smoothness checks for constant formulas
        if isinstance(formula, sp.Number):
            continue

        # For rows other than the first, check continuity and smoothness
        if i > 0:
            prev_row = df.iloc[i-1]
            prev_end_x = sp.sympify(prev_row['end_x'])
            prev_formula = sp.sympify(correct_formula(prev_row['formula']))

            # Skip if previous formula is also a constant
            if isinstance(prev_formula, sp.Number):
                continue

            if not is_continuous(prev_formula, formula, prev_end_x):
                raise ValueError(f"Discontinuity detected between row {i-1} and row {i}")

            # Check for smoothness by comparing derivatives at the boundary point
            if not is_continuous(sp.diff(prev_formula, x), sp.diff(formula, x), prev_end_x):
                raise ValueError(f"Function not smooth between row {i-1} and row {i}")

    print("CSV validation passed.")

def plot_segment(formula, start_x, end_x, ax):
    # Correct the formula syntax by adding explicit multiplication
    corrected_formula = correct_formula(formula)
    
    # Convert the corrected formula to a sympy expression
    expr = sp.sympify(corrected_formula)

    # Create a lambda function for the expression
    f = sp.lambdify(sp.Symbol('x'), expr, 'numpy')

    # Generate x values
    x_vals = np.linspace(float(start_x), float(end_x), 1000)

    # Compute y values
    y_vals = f(x_vals)

    # Plot the segment
    ax.plot(x_vals, y_vals, label=corrected_formula)

    print("plot_segment completed")

def main():
    # Read CSV and validate
    file_name = 'formula_input.csv'
    df = read_and_print_csv(file_name)
    validate_csv(df)

    fig, ax = plt.subplots()

    # Plot each segment from the CSV
    for index, row in df.iterrows():
        print(f'\n\n{row}')
        plot_segment(row['formula'], row['start_x'], row['end_x'], ax)

    # Set up plot
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()

    # Save the plot to an SVG file
    output_file = 'roller_coaster.svg'
    plt.savefig(output_file, format='svg')

    # Optionally, display the plot as well
    plt.show()

    print(f"Plot saved to {output_file}")

if __name__ == '__main__':
    main()
