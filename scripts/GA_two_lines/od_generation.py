import random
import numpy as np
import pandas as pd


def generate_field(d, save_files=True):
    """
    Generates a 2D scalar field with two Gaussian distributions over the grid.
    """

    # Create a grid
    x = np.linspace(0, d-1, 100)
    y = np.linspace(0, d-1, 100)
    X, Y = np.meshgrid(x, y)

    # Create a field
    sigma = d / 10
    center_x, center_y = d / 2, d / 2  # d / 4, d / 4
    gaussian1 = np.exp(-((X - center_x)**2 + (Y - center_y)
                       ** 2) / (2 * sigma**2))
    center_x, center_y = 3 * d / 4, 3 * d / 4
    gaussian2 = np.exp(-((X - center_x)**2 + (Y - center_y)
                       ** 2) / (2 * sigma**2))
    center_x, center_y = d / 4, 3 * d / 4
    gaussian3 = np.exp(-((X - center_x)**2 + (Y - center_y)
                       ** 2) / (2 * sigma**2))
    field = gaussian1  # + gaussian2 #+ gaussian3

    # Round field values
    field = np.round(field, 2)

    # Save field values to a CSV file
    if save_files:
        save_bus_stop_field_value_csv(X, Y, field, "data/bus_network/field.csv")

    return X, Y, field

def save_bus_stop_field_value_csv(X, Y, field, file):
    """
    Saves the field values at the bus stops to a CSV file.
    """

    # Save field values to CSV file
    with open(file, "w") as f:
        f.write("x,y,field\n")
        for i in range(len(X)):
            for j in range(len(Y)):
                f.write(f"{X[i, j]},{Y[i, j]},{field[i, j]}\n")

def OD_matrix(seed, N_B, pos, d, save_files=True):
    """
    Generates an origin-destination matrix for the bus stops based on a field.
    """

    random.seed(seed)
    np.random.seed(seed)

    # Generate field
    _, _, field = generate_field(d)

    # Compute field values at bus stops
    field_values = []
    for node in N_B:
        x, y = pos[node]
        i = int(x * 100 / d)
        j = int(y * 100 / d)
        field_values.append(field[j, i])
    field_values = np.array(field_values)

    D = np.zeros((len(N_B), len(N_B)), dtype=float)
    for i in range(len(N_B)):
        for j in range(i+1, len(N_B)):
            if i != j and random.random() < 0.2:
                D[i, j] = max(field_values[i] * 10, field_values[j] * 10)
                D[j, i] = D[i, j]

    # Round OD matrix values
    D = np.round(D, 2)

    # Mapping from node to index in the OD matrix
    node_to_bus_index = {node: idx for idx, node in enumerate(N_B)}

    # Save OD matrix to a CSV file
    if save_files:
        save_OD_matrix_csv(D, N_B, "data/bus_network/od_matrix.csv")

    return D, node_to_bus_index

def save_OD_matrix_csv(D, N_B, file):
    df = pd.DataFrame(D, index=N_B, columns=N_B)
    df.to_csv(file)