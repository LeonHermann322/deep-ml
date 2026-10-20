def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
    rows_a = len(a)
    cols_a = len(a[0]) if rows_a > 0 else -1

    rows_b = len(b)
    cols_b = len(b[0]) if rows_b > 0 else -1

    if cols_a != rows_b:
        return -1


    products = []
    for row_idx in range(rows_a):
        row = []
        for col_idx in range(cols_b):
            summed = sum([a[row_idx][i] * b[i][col_idx] for i in range(cols_a)])
            row.append(summed)
        products.append(row)

    return products
    