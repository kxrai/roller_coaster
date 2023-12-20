
import pandas as pd
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import sys


def read_and_print_csv(file_name):
    try:
        df = pd.read_csv(file_name)
        print(df)
    except Exception as e:
        print(f"Error reading the file: {e}")

# def validate_csv(df):
#     for i, row in df.iterrows():
#         # Handle 'pi' in start_x and end_x
#         start_x_str = row['start_x'].replace('pi', '*pi')
#         end_x_str = row['end_x'].replace('pi', '*pi')

#         # Evaluate start_x and end_x
#         start_x = sp.sympify(start_x_str).evalf()
#         end_x = sp.sympify(end_x_str).evalf()

#         if end_x <= start_x:
#             raise ValueError(f"End value of x must be greater than start value in row {i}")

#         # For rows other than the first, check continuity and smoothness
#         if i > 0:
#             prev_row = df.iloc[i-1]
#             prev_end_x = sp.sympify(prev_row['end_x']).evalf()
#             if start_x != prev_end_x:
#                 raise ValueError(f"Start value of x in row {i} does not match end value of x in previous row")

#             # Check for smooth transition (continuity and differentiability)
#             prev_formula = sp.sympify(prev_row['formula'])
#             if not sp.limit(formula - prev_formula, sp.Symbol('x'), prev_end_x) == 0:
#                 raise ValueError(f"Discontinuity detected between row {i-1} and row {i}")

#             if not sp.limit(sp.diff(formula) - sp.diff(prev_formula), sp.Symbol('x'), prev_end_x) == 0:
#                 raise ValueError(f"Function not smooth between row {i-1} and row {i}")

#     print("CSV validation passed.")

def validate_csv(df):
    for i, row in df.iterrows():
        # Convert start_x and end_x to strings
        start_x_str = str(row['start_x'])
        end_x_str = str(row['end_x'])

        # Check if 'pi' is present in the strings
        if 'pi' in start_x_str:
            start_x_str = start_x_str.replace('pi', '*(pi)')
        if 'pi' in end_x_str:
            end_x_str = end_x_str.replace('pi', '*(pi)')

        # Evaluate start_x and end_x using sympify with a local dictionary for pi
        local_dict = {'pi': sp.pi}
        try:
            start_x = sp.sympify(start_x_str, locals=local_dict).evalf()
            end_x = sp.sympify(end_x_str, locals=local_dict).evalf()
        except sp.SympifyError as e:
            raise ValueError(f"Error parsing start_x or end_x in row {i}: {e}")

        # Convert current formula to sympy expression and validate
        formula = sp.sympify(row['formula'])

        # For rows other than the first, check continuity and smoothness
        if i > 0:
            prev_row = df.iloc[i-1]
            prev_end_x = sp.sympify(str(prev_row['end_x']).replace('pi', 'sp.pi'), locals={'sp.pi': sp.pi}).evalf()

            if start_x != prev_end_x:
                raise ValueError(f"Start value of x in row {i} does not match end value of x in previous row")

            prev_formula = sp.sympify(prev_row['formula'])

            # Check for continuity
            if not sp.limit(formula - prev_formula, sp.Symbol('x'), prev_end_x) == 0:
                raise ValueError(f"Discontinuity detected between row {i-1} and row {i}")

            # Check for smoothness
            if not sp.limit(sp.diff(formula) - sp.diff(prev_formula), sp.Symbol('x'), prev_end_x) == 0:
                raise ValueError(f"Function not smooth between row {i-1} and row {i}")

    print("CSV validation passed.")



def plot_segment(formula, start_x, end_x, ax):
    # Convert the formula to a sympy expression
    expr = sp.sympify(formula)
    
    # Evaluate start_x and end_x to numerical values and convert to float
    start_x_num = float(sp.sympify(start_x).evalf())
    end_x_num = float(sp.sympify(end_x).evalf())

    # Create a lambda function for the expression
    f = sp.lambdify(sp.Symbol('x'), expr, 'numpy')

    # Generate x values
    x_vals = np.linspace(start_x_num, end_x_num, 1000)

    # Compute y values
    y_vals = f(x_vals)

    # Plot the segment
    ax.plot(x_vals, y_vals, label=formula)

    print("plot_segment completed")


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

def main():
    # Read and validate the CSV file
    file_name = 'lewin_input.csv'
    df = pd.read_csv(file_name)
    validate_csv(df)

    fig, ax = plt.subplots()

    # Iterate through each row and plot segments
    for index, row in df.iterrows():
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
