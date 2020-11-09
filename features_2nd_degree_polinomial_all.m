function x_polinomial = features_2nd_degree_polinomial_all(x)
    x_rows = size(x, 1);
    x_cols = size(x, 2);

    x_polinomial_cols = x_cols;
    for i=1:x_cols,
        x_polinomial_cols = x_polinomial_cols + i;
    end;
    x_polinomial = ones(x_rows, x_polinomial_cols);

    for row=1:x_rows,
        col_index = 1;
        % copy the existing values
        for col=1:x_cols,
            x_polinomial(row, col_index) = x(row, col);
            col_index = col_index + 1;
        end;
        % add 2nd degree polinomial values
        for col_i=1:x_cols,
            for col_j=col_i:x_cols,
                x_polinomial(row, col_index) = x(row, col_i) * x(row, col_j);
                col_index = col_index + 1;
            end;
        end;
    end;
end