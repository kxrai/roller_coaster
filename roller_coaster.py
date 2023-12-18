import pandas as pd
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import sys

# def read_csv(file_name):
#     return pd.read_csv(file_name)


def read_and_print_csv(file_name):
    try:
        df = pd.read_csv(file_name)
        print(df)
    except Exception as e:
        print(f"Error reading the file: {e}")

file_name = 'formula_input.csv'  # Just the file name
read_and_print_csv(file_name)

def validate_csv(df):
    for i, row in df.iterrows():
        # Convert formulas to sympy expressions and validate
        try:
            formula = sp.sympify(row['formula'])
        except sp.SympifyError:
            raise ValueError(f"Invalid formula in row {i}: {row['formula']}")

        # Check if end_x is greater than start_x
        start_x = sp.sympify(row['start_x'])
        end_x = sp.sympify(row['end_x'])
        if end_x <= start_x:
            raise ValueError(f"End value of x must be greater than start value in row {i}")

        # For rows other than the first, check continuity and smoothness
        if i > 0:
            prev_row = df.iloc[i-1]
            prev_end_x = sp.sympify(prev_row['end_x'])
            if start_x != prev_end_x:
                raise ValueError(f"Start value of x in row {i} does not match end value of x in previous row")

            # Check for smooth transition (continuity and differentiability)
            prev_formula = sp.sympify(prev_row['formula'])
            if not sp.limit(formula - prev_formula, sp.Symbol('x'), prev_end_x) == 0:
                raise ValueError("Discontinuity detected between row {i-1} and row {i}")

            if not sp.limit(sp.diff(formula) - sp.diff(prev_formula), sp.Symbol('x'), prev_end_x) == 0:
                raise ValueError("Function not smooth between row {i-1} and row {i}")

    print("CSV validation passed.")


#-------------------------------------------------------------------------------------------
# def plot_segment(formula, start_x, end_x, ax):
#     # Convert the formula to a sympy expression
#     expr = sp.sympify(formula)

#     # Create a lambda function for the expression
#     f = sp.lambdify(sp.Symbol('x'), expr, 'numpy')

#     # Generate x values
#     x_vals = np.linspace(float(start_x), float(end_x), 1000)

#     # Compute y values
#     y_vals = f(x_vals)

#     # Plot the segment
#     ax.plot(x_vals, y_vals, label=formula)

#     print("plot_segment completed")
    
def plot_segment(formula, start_x, end_x, ax):
    # Check if the formula is a constant number
    try:
        # Convert the formula to a sympy expression
        expr = sp.sympify(formula)

        if isinstance(expr, sp.Number):
            # It's a constant, so plot a horizontal line
            ax.hlines(float(expr), float(start_x), float(end_x), label=f'y={expr}')
        else:
            # Create a lambda function for the expression
            f = sp.lambdify(sp.Symbol('x'), expr, 'numpy')
            # Generate x values
            x_vals = np.linspace(float(start_x), float(end_x), 1000)
            # Compute y values and plot
            y_vals = f(x_vals)
            ax.plot(x_vals, y_vals, label=str(formula))

    except Exception as e:
        raise ValueError(f"Error in plotting segment: {e}")

    print(f"plot_segment for {formula} completed")

#-------------------------------------------------------------------------------------------
# def generate_roller_coaster_plot(df, output_file):
#     # Generate the complete roller coaster plot
#     pass

# def main():
#     fig, ax = plt.subplots()

#     # Plot segments
#     plot_segment('x**2', -2, 2, ax)
#     plot_segment('sin(x)', 2, 5, ax)

#     # Set up plot
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.legend()

#     # Save the plot to an SVG file
#     output_file = 'roller_coaster.svg'
#     plt.savefig(output_file, format='svg')

#     # Optionally, display the plot as well
#     plt.show()

#     print(f"Plot saved to {output_file}")

# if __name__ == '__main__':
#     main()

def generate_roller_coaster_plot(df, output_file, ax):
    # Iterate over the dataframe and plot each segment
    for index, row in df.iterrows():
        plot_segment(row['formula'], row['start_x'], row['end_x'], ax)
    
    # Set up plot
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    
    # Save the plot to an SVG file
    plt.savefig(output_file, format='svg')
    print(f"Plot saved to {output_file}")

    # Optionally, display the plot as well
    plt.show()

def main():
    # Read the CSV file into a DataFrame
    df = pd.read_csv('formula_input.csv')
    
    # Validate the CSV data
    validate_csv(df)
    
    # Prepare the figure for plotting
    fig, ax = plt.subplots()
    
    # Generate the roller coaster plot
    output_file = 'roller_coaster.svg'
    generate_roller_coaster_plot(df, output_file, ax)

if __name__ == '__main__':
    main()
