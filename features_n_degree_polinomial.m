function x_polinomial = features_n_degree_polinomial(x, degree)
    x_rows = size(x, 1);
    x_cols = size(x, 2);
    x_polinomial = ones(x_rows, x_cols * degree);

    for row=1:x_rows,
        col_index = 1;
        for pow=1:degree,
            for col=1:x_cols,
                x_polinomial(row, col_index) = x(row, col)^pow;
                col_index = col_index + 1;
            end;
        end;
    end;
end