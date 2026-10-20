def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    """
    Transpose a 2D matrix by swapping rows and columns.
    
    Args:
        a: A 2D matrix of shape (m, n)
    
    Returns:
        The transposed matrix of shape (n, m)
    """
    row_count = len(a)

    if row_count == 0:
        return []

    column_count = len(a[0])

    transposed = []
    for i in range(column_count):
        new_row = []
        for j in range(row_count):
            new_row.append(a[j][i])
        transposed.append(new_row)

    return transposed