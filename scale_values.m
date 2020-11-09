function x = scale_values(x)
    % prepare counts (rows and cols), mean and std
    [x_rows, x_cols] = size(x);
    x_mean = mean(x);
    x_std = std(x);

    % scale values
    for row=1:x_rows,
        for col=1:x_cols,
            x(row, col) = (x(row, col) - x_mean(col)) / x_std(col);
        end;
    end;
end